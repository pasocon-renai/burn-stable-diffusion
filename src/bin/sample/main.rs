use stablediffusion::{
    model::stablediffusion::{load::load_stable_diffusion, *},
    tokenizer::SimpleTokenizer,
};

use burn::{
    module::Module,
    tensor::backend::Backend,
};

use burn::record::{self, NamedMpkFileRecorder, FullPrecisionSettings, Recorder};

use std::process;

fn load_stable_diffusion_model_file<B: Backend>(
    filename: &str,
    device: &B::Device,
) -> Result<StableDiffusion<B>, record::RecorderError> {
    let r=NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(filename.into(), device)
        .map(|record| StableDiffusionConfig::new().init(device).load_record(record))?;
    //NamedMpkFileRecorder::<FullPrecisionSettings>::new().record(r.clone().into_record(),filename.into());

    Ok(r)
}


fn _main<B:Backend>(model_type:&str,model_name:&str,unconditional_guidance_scale:f64,n_steps:usize,prompt:&str,output_image_name:&str){
    let device=Default::default();
    println!("Loading tokenizer...");
    let tokenizer = SimpleTokenizer::new().unwrap();
    println!("Loading model...");
    let sd: StableDiffusion<B> = if model_type == "burn" {
        load_stable_diffusion_model_file(model_name, &device).unwrap_or_else(|err| {
            eprintln!("Error loading model: {}", err);
            process::exit(1);
        })
    } else {
        load_stable_diffusion(model_name, &device).unwrap_or_else(|err| {
            eprintln!("Error loading model dump: {}", err);
            process::exit(1);
        })
    };

    let unconditional_context = sd.unconditional_context(&tokenizer);
    let context = sd.context(&tokenizer, prompt).unsqueeze::<3>(); //.repeat(0, 2); // generate 2 samples

    println!("Sampling image...");
    let images = sd.sample_image(
        context,
        unconditional_context,
        unconditional_guidance_scale,
        n_steps,
    );
    save_images(&images, output_image_name, 512, 512).unwrap_or_else(|err| {
        eprintln!("Error saving image: {}", err);
        process::exit(1);
    });
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 && args.len() != 8 {
        eprintln!("Usage: {} <model_type(burn or dump)> <model_name> <unconditional_guidance_scale> <n_diffusion_steps> <prompt> <output_image_name> [device(cuda, mps, cpu)]", args[0]);
        process::exit(1);
    }

    let model_type = &args[1];
    let model_name = &args[2];
    let unconditional_guidance_scale: f64 = args[3].parse().unwrap_or_else(|_| {
        eprintln!("Error: Invalid unconditional guidance scale.");
        process::exit(1);
    });
    let n_steps: usize = args[4].parse().unwrap_or_else(|_| {
        eprintln!("Error: Invalid number of diffusion steps.");
        process::exit(1);
    });
    let prompt = &args[5];
    let output_image_name = &args[6];

    // Optional device parameter
    let device_arg = if args.len() == 8 { Some(&args[7]) } else { None };

    if let Some(dev_str)=device_arg&&dev_str=="cpu"{
        _main::<burn::backend::Flex>(model_type,model_name,unconditional_guidance_scale,n_steps,prompt,output_image_name);
    }else{
        _main::<burn::backend::Wgpu>(model_type,model_name,unconditional_guidance_scale,n_steps,prompt,output_image_name);
    }
}

use image::{self, ColorType::Rgb8, ImageResult};

fn save_images(images: &Vec<Vec<u8>>, basepath: &str, width: u32, height: u32) -> ImageResult<()> {
    for (index, img_data) in images.iter().enumerate() {
        let path = format!("{}{}.png", basepath, index);
        image::save_buffer(path, &img_data[..], width, height, Rgb8)?;
    }

    Ok(())
}

// save red test image
#[allow(dead_code)]
fn save_test_image() -> ImageResult<()> {
    let width = 256;
    let height = 256;
    let raw: Vec<_> = (0..width * height)
        .into_iter()
        .flat_map(|i| {
            let row = i / width;
            let red = (255.0 * row as f64 / height as f64) as u8;

            [red, 0, 0]
        })
        .collect();

    image::save_buffer("red.png", &raw[..], width, height, Rgb8)
}
