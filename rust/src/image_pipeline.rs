use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use fast_image_resize::{
    FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer, images::Image as FirImage,
};
use image::RgbImage;
use numpy::{IntoPyArray, PyArray4, ndarray::Array4};
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use thiserror::Error;

static DATA_POOLS: OnceLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();

#[derive(Clone, Debug)]
struct TransformConfig {
    height: usize,
    width: usize,
    mean: [f32; 3],
    std: [f32; 3],
    training: bool,
    flip_prob: f32,
    color_jitter: [f32; 4],
    crop_padding: usize,
    erase_prob: f32,
    erase_scale: [f32; 2],
    erase_ratio: [f32; 2],
    grayscale_prob: f32,
}

#[derive(Debug, Error)]
enum ImagePipelineError {
    #[error("failed to read image {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to decode image {path}: {source}")]
    Decode {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
    #[error("failed to process image {path}: {message}")]
    Process { path: PathBuf, message: String },
}

#[pyfunction]
#[pyo3(signature = (
    paths,
    batch_seed,
    height,
    width,
    mean,
    std,
    training,
    flip_prob,
    color_jitter,
    crop_padding,
    erase_prob,
    erase_scale,
    erase_ratio,
    grayscale_prob,
    threads=1
))]
#[allow(clippy::too_many_arguments)]
pub fn load_image_batch<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    batch_seed: u64,
    height: usize,
    width: usize,
    mean: Vec<f32>,
    std: Vec<f32>,
    training: bool,
    flip_prob: f32,
    color_jitter: Vec<f32>,
    crop_padding: usize,
    erase_prob: f32,
    erase_scale: Vec<f32>,
    erase_ratio: Vec<f32>,
    grayscale_prob: f32,
    threads: usize,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    if paths.is_empty() {
        return Err(PyValueError::new_err(
            "paths must contain at least one image",
        ));
    }
    let config = validate_config(
        height,
        width,
        mean,
        std,
        training,
        flip_prob,
        color_jitter,
        crop_padding,
        erase_prob,
        erase_scale,
        erase_ratio,
        grayscale_prob,
    )?;
    if threads == 0 {
        return Err(PyValueError::new_err("threads must be positive"));
    }

    let paths: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
    let image_len = 3 * config.height * config.width;
    let mut output = vec![0.0_f32; paths.len() * image_len];
    let result = py.detach(|| {
        if threads == 1 {
            output
                .chunks_mut(image_len)
                .zip(paths.iter())
                .enumerate()
                .try_for_each(|(index, (destination, path))| {
                    process_image(path, destination, &config, derived_seed(batch_seed, index))
                })
        } else {
            let pool = data_pool(threads).map_err(|message| ImagePipelineError::Process {
                path: PathBuf::from("<rayon>"),
                message,
            })?;
            pool.install(|| {
                output
                    .par_chunks_mut(image_len)
                    .zip(paths.par_iter())
                    .enumerate()
                    .try_for_each(|(index, (destination, path))| {
                        process_image(path, destination, &config, derived_seed(batch_seed, index))
                    })
            })
        }
    });
    result.map_err(image_error_to_python)?;

    let array = Array4::from_shape_vec((paths.len(), 3, config.height, config.width), output)
        .map_err(|error| {
            PyRuntimeError::new_err(format!("failed to shape native image batch: {error}"))
        })?;
    Ok(array.into_pyarray(py))
}

#[allow(clippy::too_many_arguments)]
fn validate_config(
    height: usize,
    width: usize,
    mean: Vec<f32>,
    std: Vec<f32>,
    training: bool,
    flip_prob: f32,
    color_jitter: Vec<f32>,
    crop_padding: usize,
    erase_prob: f32,
    erase_scale: Vec<f32>,
    erase_ratio: Vec<f32>,
    grayscale_prob: f32,
) -> PyResult<TransformConfig> {
    if height == 0 || width == 0 {
        return Err(PyValueError::new_err("height and width must be positive"));
    }
    let mean = vector3(mean, "mean")?;
    let std = vector3(std, "std")?;
    if mean
        .iter()
        .chain(std.iter())
        .any(|value| !value.is_finite())
    {
        return Err(PyValueError::new_err(
            "mean and std must contain only finite values",
        ));
    }
    if std.iter().any(|value| *value <= 0.0) {
        return Err(PyValueError::new_err("std values must be positive"));
    }
    require_probability(flip_prob, "flip_prob")?;
    require_probability(erase_prob, "erase_prob")?;
    require_probability(grayscale_prob, "grayscale_prob")?;
    let color_jitter = vector4(color_jitter, "color_jitter")?;
    if color_jitter[..3]
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || !color_jitter[3].is_finite()
        || !(0.0..=0.5).contains(&color_jitter[3])
    {
        return Err(PyValueError::new_err(
            "color_jitter brightness/contrast/saturation must be non-negative and hue must be in [0, 0.5]",
        ));
    }
    let erase_scale = vector2(erase_scale, "erase_scale")?;
    let erase_ratio = vector2(erase_ratio, "erase_ratio")?;
    if erase_scale.iter().any(|value| !value.is_finite())
        || erase_scale[0] <= 0.0
        || erase_scale[0] > erase_scale[1]
        || erase_scale[1] > 1.0
    {
        return Err(PyValueError::new_err(
            "erase_scale must satisfy 0 < min <= max <= 1",
        ));
    }
    if erase_ratio.iter().any(|value| !value.is_finite())
        || erase_ratio[0] <= 0.0
        || erase_ratio[0] > erase_ratio[1]
    {
        return Err(PyValueError::new_err(
            "erase_ratio must satisfy 0 < min <= max",
        ));
    }
    Ok(TransformConfig {
        height,
        width,
        mean,
        std,
        training,
        flip_prob,
        color_jitter,
        crop_padding,
        erase_prob,
        erase_scale,
        erase_ratio,
        grayscale_prob,
    })
}

fn vector2(values: Vec<f32>, name: &str) -> PyResult<[f32; 2]> {
    values.try_into().map_err(|values: Vec<f32>| {
        PyValueError::new_err(format!(
            "{name} must contain 2 values, got {}",
            values.len()
        ))
    })
}

fn vector3(values: Vec<f32>, name: &str) -> PyResult<[f32; 3]> {
    values.try_into().map_err(|values: Vec<f32>| {
        PyValueError::new_err(format!(
            "{name} must contain 3 values, got {}",
            values.len()
        ))
    })
}

fn vector4(values: Vec<f32>, name: &str) -> PyResult<[f32; 4]> {
    values.try_into().map_err(|values: Vec<f32>| {
        PyValueError::new_err(format!(
            "{name} must contain 4 values, got {}",
            values.len()
        ))
    })
}

fn require_probability(value: f32, name: &str) -> PyResult<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(PyValueError::new_err(format!("{name} must be in [0, 1]")));
    }
    Ok(())
}

fn data_pool(threads: usize) -> Result<Arc<ThreadPool>, String> {
    let pools = DATA_POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = pools
        .lock()
        .map_err(|_| "native data thread-pool registry is poisoned".to_owned())?;
    if let Some(pool) = guard.get(&threads) {
        return Ok(Arc::clone(pool));
    }
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("t2c-image-{index}"))
            .build()
            .map_err(|error| format!("failed to create {threads}-thread data pool: {error}"))?,
    );
    guard.insert(threads, Arc::clone(&pool));
    Ok(pool)
}

fn derived_seed(batch_seed: u64, index: usize) -> u64 {
    let mut value = batch_seed.wrapping_add((index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn process_image(
    path: &Path,
    destination: &mut [f32],
    config: &TransformConfig,
    seed: u64,
) -> Result<(), ImagePipelineError> {
    let bytes = std::fs::read(path).map_err(|source| ImagePipelineError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let mut image = image::load_from_memory(&bytes)
        .map_err(|source| ImagePipelineError::Decode {
            path: path.to_path_buf(),
            source,
        })?
        .to_rgb8();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    if config.training {
        if rng.random::<f32>() < config.flip_prob {
            image::imageops::flip_horizontal_in_place(&mut image);
        }
        apply_color_jitter(&mut image, config.color_jitter, &mut rng);
        // Drawn only when enabled so disabled grayscale keeps the pre-existing
        // per-image random stream bitwise unchanged.
        if config.grayscale_prob > 0.0 && rng.random::<f32>() < config.grayscale_prob {
            to_grayscale(&mut image);
        }
    }
    let image = resize_rgb(image, config.width as u32, config.height as u32, path)?;
    write_normalized_chw(&image, destination, config, &mut rng);
    if config.training {
        apply_random_erasing(destination, config, &mut rng);
    }
    Ok(())
}

fn resize_rgb(
    image: RgbImage,
    width: u32,
    height: u32,
    path: &Path,
) -> Result<RgbImage, ImagePipelineError> {
    let source_width = image.width();
    let source_height = image.height();
    let source = FirImage::from_vec_u8(
        source_width,
        source_height,
        image.into_raw(),
        PixelType::U8x3,
    )
    .map_err(|error| ImagePipelineError::Process {
        path: path.to_path_buf(),
        message: format!("invalid decoded RGB buffer: {error}"),
    })?;
    let mut destination = FirImage::new(width, height, PixelType::U8x3);
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));
    Resizer::new()
        .resize(&source, &mut destination, &options)
        .map_err(|error| ImagePipelineError::Process {
            path: path.to_path_buf(),
            message: format!("bilinear resize failed: {error}"),
        })?;
    RgbImage::from_raw(width, height, destination.into_vec()).ok_or_else(|| {
        ImagePipelineError::Process {
            path: path.to_path_buf(),
            message: "resized RGB buffer has an invalid length".to_owned(),
        }
    })
}

fn apply_color_jitter(image: &mut RgbImage, values: [f32; 4], rng: &mut ChaCha8Rng) {
    let brightness = rng.random_range((1.0 - values[0]).max(0.0)..=1.0 + values[0]);
    let contrast = rng.random_range((1.0 - values[1]).max(0.0)..=1.0 + values[1]);
    let saturation = rng.random_range((1.0 - values[2]).max(0.0)..=1.0 + values[2]);
    let hue = rng.random_range(-values[3]..=values[3]);
    let mut operations = [0_u8, 1, 2, 3];
    operations.shuffle(rng);
    for operation in operations {
        match operation {
            0 => adjust_brightness(image, brightness),
            1 => adjust_contrast(image, contrast),
            2 => adjust_saturation(image, saturation),
            3 => adjust_hue(image, hue),
            _ => unreachable!(),
        }
    }
}

fn adjust_brightness(image: &mut RgbImage, factor: f32) {
    for pixel in image.pixels_mut() {
        for channel in &mut pixel.0 {
            *channel = to_u8(*channel as f32 * factor);
        }
    }
}

fn adjust_contrast(image: &mut RgbImage, factor: f32) {
    let pixel_count = image.width() as f32 * image.height() as f32;
    let mean = image.pixels().map(|pixel| luminance(pixel.0)).sum::<f32>() / pixel_count;
    for pixel in image.pixels_mut() {
        for channel in &mut pixel.0 {
            *channel = to_u8(mean + (*channel as f32 - mean) * factor);
        }
    }
}

fn adjust_saturation(image: &mut RgbImage, factor: f32) {
    for pixel in image.pixels_mut() {
        let gray = luminance(pixel.0);
        for channel in &mut pixel.0 {
            *channel = to_u8(gray + (*channel as f32 - gray) * factor);
        }
    }
}

fn adjust_hue(image: &mut RgbImage, factor: f32) {
    if factor == 0.0 {
        return;
    }
    for pixel in image.pixels_mut() {
        let [red, green, blue] = pixel.0.map(|value| value as f32 / 255.0);
        let max = red.max(green).max(blue);
        let min = red.min(green).min(blue);
        let delta = max - min;
        if delta == 0.0 {
            continue;
        }
        let base_hue = if max == red {
            ((green - blue) / delta).rem_euclid(6.0)
        } else if max == green {
            (blue - red) / delta + 2.0
        } else {
            (red - green) / delta + 4.0
        } / 6.0;
        let hue = (base_hue + factor).rem_euclid(1.0);
        let saturation = if max == 0.0 { 0.0 } else { delta / max };
        pixel.0 = hsv_to_rgb(hue, saturation, max);
    }
}

fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> [u8; 3] {
    let scaled = hue * 6.0;
    let sector = scaled.floor() as i32;
    let fraction = scaled - sector as f32;
    let p = value * (1.0 - saturation);
    let q = value * (1.0 - fraction * saturation);
    let t = value * (1.0 - (1.0 - fraction) * saturation);
    let (red, green, blue) = match sector.rem_euclid(6) {
        0 => (value, t, p),
        1 => (q, value, p),
        2 => (p, value, t),
        3 => (p, q, value),
        4 => (t, p, value),
        _ => (value, p, q),
    };
    [
        to_u8(red * 255.0),
        to_u8(green * 255.0),
        to_u8(blue * 255.0),
    ]
}

fn to_grayscale(image: &mut RgbImage) {
    for pixel in image.pixels_mut() {
        let gray = to_u8(luminance(pixel.0));
        pixel.0 = [gray; 3];
    }
}

fn luminance(rgb: [u8; 3]) -> f32 {
    0.299 * rgb[0] as f32 + 0.587 * rgb[1] as f32 + 0.114 * rgb[2] as f32
}

fn to_u8(value: f32) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

fn write_normalized_chw(
    image: &RgbImage,
    destination: &mut [f32],
    config: &TransformConfig,
    rng: &mut ChaCha8Rng,
) {
    let plane = config.height * config.width;
    let (crop_x, crop_y) = if config.training && config.crop_padding > 0 {
        (
            rng.random_range(0..=2 * config.crop_padding),
            rng.random_range(0..=2 * config.crop_padding),
        )
    } else {
        (config.crop_padding, config.crop_padding)
    };
    for output_y in 0..config.height {
        for output_x in 0..config.width {
            let source_x = output_x as isize + crop_x as isize - config.crop_padding as isize;
            let source_y = output_y as isize + crop_y as isize - config.crop_padding as isize;
            let rgb = if source_x >= 0
                && source_x < config.width as isize
                && source_y >= 0
                && source_y < config.height as isize
            {
                image.get_pixel(source_x as u32, source_y as u32).0
            } else {
                [0_u8; 3]
            };
            let offset = output_y * config.width + output_x;
            for channel in 0..3 {
                destination[channel * plane + offset] =
                    (rgb[channel] as f32 / 255.0 - config.mean[channel]) / config.std[channel];
            }
        }
    }
}

fn apply_random_erasing(destination: &mut [f32], config: &TransformConfig, rng: &mut ChaCha8Rng) {
    if rng.random::<f32>() >= config.erase_prob {
        return;
    }
    let area = (config.height * config.width) as f32;
    let log_min = config.erase_ratio[0].ln();
    let log_max = config.erase_ratio[1].ln();
    for _ in 0..10 {
        let target_area = area * rng.random_range(config.erase_scale[0]..=config.erase_scale[1]);
        let aspect_ratio = rng.random_range(log_min..=log_max).exp();
        let erase_height = (target_area * aspect_ratio).sqrt().round() as usize;
        let erase_width = (target_area / aspect_ratio).sqrt().round() as usize;
        if erase_height < config.height && erase_width < config.width {
            let top = rng.random_range(0..=config.height - erase_height);
            let left = rng.random_range(0..=config.width - erase_width);
            let plane = config.height * config.width;
            for channel in 0..3 {
                for y in top..top + erase_height {
                    let start = channel * plane + y * config.width + left;
                    destination[start..start + erase_width].fill(0.0);
                }
            }
            return;
        }
    }
}

fn image_error_to_python(error: ImagePipelineError) -> PyErr {
    match error {
        ImagePipelineError::Read { .. } | ImagePipelineError::Decode { .. } => {
            PyOSError::new_err(error.to_string())
        }
        ImagePipelineError::Process { .. } => PyRuntimeError::new_err(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::{RgbImage, derived_seed, hsv_to_rgb, luminance, to_grayscale};

    #[test]
    fn derived_seeds_are_stable_and_distinct() {
        assert_eq!(derived_seed(7, 0), derived_seed(7, 0));
        assert_ne!(derived_seed(7, 0), derived_seed(7, 1));
    }

    #[test]
    fn grayscale_hue_rotation_is_stable() {
        assert_eq!(hsv_to_rgb(0.0, 0.0, 0.5), [128, 128, 128]);
        assert_eq!(luminance([255, 255, 255]).round() as u8, 255);
    }

    #[test]
    fn grayscale_replicates_luminance_across_channels() {
        let mut image = RgbImage::from_raw(2, 1, vec![255, 0, 0, 10, 200, 30]).unwrap();
        to_grayscale(&mut image);
        assert_eq!(image.get_pixel(0, 0).0, [76, 76, 76]);
        assert_eq!(image.get_pixel(1, 0).0, [124, 124, 124]);
    }
}
