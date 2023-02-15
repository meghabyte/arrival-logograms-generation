## Overview
*Train a text-to-image model to generate Arrival-inspired logograms for novel concepts!*

[Arrival](https://en.wikipedia.org/wiki/Arrival_(film)) is one of my favorite science fiction movies, particularly due to its focus on the interactive nature of language. As shown below, the logograms designed for the alien language in the film are beautiful smoke-like circles, and are even released in [another GitHub repository](https://github.com/WolframResearch/Arrival-Movie-Live-Coding) with additional analysis from Wolfram Research! 

<figure>
  <img
  width="414"
  src="https://user-images.githubusercontent.com/5402873/218632617-45b4623b-c93f-4898-a617-41310d8021a5.png"
  alt="train_examples.">
</figure>

For a while, I have wanted a tattoo of one of the logograms, but the officially released logograms mainly cover concepts like `weapon`, `ship grounded`, and `there is no linear time`, which, while relevant for a science fiction film, don't feel particularly personal or compelling for something like a permamanent tattoo.  So, I decided to fine-tune [Stable Diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion), a text-to-image diffusion model, over the 38 logograms (small dataset!) released by Wolfram so that I could generate logograms for novel concepts that I'd be more willing to permanantly inked on me :stuck_out_tongue: I ended up getting a tattoo of a logogram generated for the concept `resilience` above my right elbow, and I describe the process below!

## Model Training Instructions
Here are the following steps to repeat the overall process, which come from [here](https://huggingface.co/docs/diffusers/training/text2image). 

1. Install HuggingFace `diffusers`, which provides a variety of pretrained vision models, from source: `pip install git+https://github.com/huggingface/diffusers`

2. Install the `accelerate` library: `pip install accelerate`

3. Install all dependencies: `pip install -U -r requirements.txt`

4. Verify all training data is located in the `data/train/` directory. To add more logograms, place the image in this folder and update the file `data/train/metadata.jsonl`, which contains the mapping between the filename and text caption. 


5. Run the following command to finetune a Stable Diffusion v1-4 model: 

```
accelerate launch train.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --train_data_dir="./data" \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=50 \
  --learning_rate=1e-03 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="model"
  ```

In general, I found training for 30-50 steps most reasonable, with a learning rate of 1e-03. I had hoped training for more steps would make the model generate higher-contrast samples that are even more similar to the training examples (due to overfitting), but training longer led to a lot more complex threads and "splatters" around the circles, creating a messier look. Training for less leads to simpler patterns, but the background texture is not plain white, shown next.   

## Generating Logogram Samples
After fine-tuning has finished, run `python generate.py` to generate samples. Modify the `PROMPTS` variable to be the list of the text concepts to generate concepts for. The script will generate 20 samples per prompt. 

In general, the generated logograms aren't on a clean white background, and have many artifacts, but I think they capture the overall circular shape and "smokiness" aesthetic quite well. I quickly cleaned them up using the Magic Wand tool in Photoshop. Here are the original generated logograms with the their cleaned version for different  unseen concepts!     
<img width="420" alt="combined" src="https://user-images.githubusercontent.com/5402873/218630765-00e91c7b-8d8d-419c-b814-ba969fbd1a9f.png">

### Tattoo
I ended up picking the concept ``resilience`` for my tattoo, which feels a lot more meaningful and personal than the existing logograms. The tattoo artist said that some of the smoky lines will merge together at the size I wanted the tattoo, so he drew his own version inspired by the generated version. Here is what it eventually looked like! :blush:


<img width="300" alt="tattoo" src="https://user-images.githubusercontent.com/5402873/218891039-ba2f17a0-8cee-44d8-8e5d-d6fce445d853.png">




