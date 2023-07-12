
import torch
from PIL import Image
from open_flamingo import create_model_and_transforms, create_model_and_transforms_v1, prepare_model_for_tuning, get_params_count_summary, resume_from_checkpoints

class Inferencer:
    def __init__(
            self, 
            checkpoint_paths,
            lm_path,
            tuning_config,
            clip_vision_encoder_path,
            clip_vision_encoder_pretrained='openai',
            cross_attn_every_n_layers=4,
            v1=False
            ):
        self.v1 = v1
        if self.v1:
            print('using v1 model')
            model, image_processor, tokenizer = create_model_and_transforms_v1(
                clip_vision_encoder_path="ViT-L-14-336",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path=lm_path,
                tokenizer_path=lm_path,
                cross_attn_every_n_layers=cross_attn_every_n_layers,
                lora_weights=tuning_config,
            )
            for checkpoint_path in checkpoint_paths.split(','):
                print(f'loading {checkpoint_path}')
                checkpoint = torch.load(checkpoint_path, map_location="cpu")

                if 'model_state_dict' in checkpoint.keys():
                    print(f'removing "model_state_dict" from checkpoint {checkpoint_path}')
                    checkpoint = checkpoint['model_state_dict']

                if next(iter(checkpoint.items()))[0].startswith('module'):
                    print(f'removing "module" from checkpoint {checkpoint_path}')
                    checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}
                    
                message = model.load_state_dict(checkpoint, strict=False)
        else:
            model, image_processor, tokenizer = create_model_and_transforms(
                clip_vision_encoder_path=clip_vision_encoder_path,
                clip_vision_encoder_pretrained=clip_vision_encoder_pretrained,
                lang_encoder_path=lm_path,
                tokenizer_path=lm_path,
                cross_attn_every_n_layers=cross_attn_every_n_layers,
            )
            model, checkpoint, resume_from_epoch, message = resume_from_checkpoints(model, checkpoint_paths)
            model, config = prepare_model_for_tuning(model, tuning_config)
            if (config['from_pretrained'] or config['lora']):
                model, checkpoint, resume_from_epoch, message = resume_from_checkpoints(model, checkpoint_paths)
        
        # print(get_params_count_summary(model))
        model.half()
        model = model.to("cuda")
        model.eval()
        tokenizer.padding_side = "left"
        tokenizer.add_eos_token = False
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, 
        prompt, 
        images, 
        max_new_token=1024, 
        num_beams=3, 
        temperature=1.0,
        top_k=20, 
        top_p=0.9, 
        do_sample=True, 
        length_penalty=1.0, 
        no_repeat_ngram_size=3,
        response_split="### Assistant:"
    ):
        # Ensure prompts is a list
        if isinstance(prompt, str):
            prompt = [prompt]

        # Ensure images is a two-dimensional list
        if len(images) == 0 or isinstance(images[0], str):
            images = [images]

        # Convert prompts to input tensors
        prompt = self.tokenizer(prompt, return_tensors="pt")

        # Load and preprocess images
        processed_images = []
        for image_paths in images:
            if len(image_paths) == 0:
                image_paths = [Image.new('RGB', (224, 224), color='black')]
            else:
                image_paths = [Image.open(fp) for fp in image_paths]

            vision_x = [self.image_processor(im).unsqueeze(0).unsqueeze(0) for im in image_paths] # Add an extra dimension for T_img
            vision_x = torch.cat(vision_x, dim=0) # Concatenate along the T_img dimension
            processed_images.append(vision_x)

        vision_x = torch.stack(processed_images).half().cuda() 

        # Generate output
        with torch.no_grad():
            if self.v1:
                output_ids = self.model.generate(
                    vision_x=vision_x,
                    lang_x=prompt["input_ids"].cuda(),
                    attention_mask=prompt["attention_mask"].cuda(),
                    max_new_tokens=max_new_token,
                    num_beams=num_beams,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )
            else:
                output_ids = self.model.generate(
                    vision_x=vision_x,
                    lang_x=prompt["input_ids"].cuda(),
                    attention_mask=prompt["attention_mask"].cuda(),
                    max_new_tokens=max_new_token,
                    num_beams=num_beams,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    eos_token_id=self.tokenizer.eos_token_id
                )

        # Decode and print the generated texts
        generated_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        results = []
        for text in generated_texts:
            print(text)
            result = text.split(response_split)[-1].strip()
            results.append(result)
            
        return results, generated_texts


if __name__=='__main__':
    pass