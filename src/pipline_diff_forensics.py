import argparse
import numpy as np
from PIL import Image
import PIL
import os 
from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionDiffEditPipeline, DDIMInverseScheduler
from diffusers.utils import BaseOutput
import torch
from typing import Optional, Callable, Union, List, Dict, Any
import os
from dataclasses import dataclass

@dataclass
class DiffEditInversionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        latents (`torch.FloatTensor`)
            inverted latents tensor
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `num_timesteps * batch_size` or numpy array of shape `(num_timesteps,
            batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the
            diffusion pipeline.
    """

    latents: torch.FloatTensor
    images: Union[List[PIL.Image.Image], np.ndarray]
    noise_features: Optional[torch.Tensor]

class DiffForensicPipeline(StableDiffusionDiffEditPipeline):
    def decode_latents(self, latents, return_type):
        with torch.no_grad():
            latents = 1 / self.vae.config.scaling_factor * latents
            image = self.vae.decode(latents, return_dict=False)[0]
            image = (image / 2 + 0.5).clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16

            image = image.cpu().permute(0, 2, 3, 1).float()
            if return_type == "np":
                image = image.numpy()

            return image
    
    @torch.no_grad()
    def invert(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        num_inference_steps: int = 50,
        inpaint_strength: float = 0.8,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        decode_latents: bool = False,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        lambda_auto_corr: float = 20.0,
        lambda_kl: float = 20.0,
        num_reg_steps: int = 0,
        num_auto_corr_rolls: int = 5,
        return_noise=False
    ):
        r"""
        Generate inverted latents given a prompt and image.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`PIL.Image.Image`):
                `Image` or tensor representing an image batch to produce the inverted latents guided by `prompt`.
            inpaint_strength (`float`, *optional*, defaults to 0.8):
                Indicates extent of the noising process to run latent inversion. Must be between 0 and 1. When
                `inpaint_strength` is 1, the inversion process is run for the full number of iterations specified in
                `num_inference_steps`. `image` is used as a reference for the inversion process, and adding more noise
                increases `inpaint_strength`. If `inpaint_strength` is 0, no inpainting occurs.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            decode_latents (`bool`, *optional*, defaults to `False`):
                Whether or not to decode the inverted latents into a generated image. Setting this argument to `True`
                decodes all inverted latents for each timestep into a list of generated images.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.DiffEditInversionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the
                [`~models.attention_processor.AttnProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            lambda_auto_corr (`float`, *optional*, defaults to 20.0):
                Lambda parameter to control auto correction.
            lambda_kl (`float`, *optional*, defaults to 20.0):
                Lambda parameter to control Kullback-Leibler divergence output.
            num_reg_steps (`int`, *optional*, defaults to 0):
                Number of regularization loss steps.
            num_auto_corr_rolls (`int`, *optional*, defaults to 5):
                Number of auto correction roll steps.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.pipeline_stable_diffusion_diffedit.DiffEditInversionPipelineOutput`] or
            `tuple`:
                If `return_dict` is `True`,
                [`~pipelines.stable_diffusion.pipeline_stable_diffusion_diffedit.DiffEditInversionPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is the inverted latents tensors
                ordered by increasing noise, and the second is the corresponding decoded images if `decode_latents` is
                `True`, otherwise `None`.
        """

        # 1. Check inputs
        self.check_inputs(
            prompt,
            inpaint_strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        if image is None:
            raise ValueError("`image` input cannot be undefined.")

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        
        noise_features = []
        with torch.no_grad():
            # 3. Preprocess image
            image = self.image_processor.preprocess(image)

            # 4. Prepare latent variables
            num_images_per_prompt = 1
            latents = self.prepare_image_latents(
                image, batch_size * num_images_per_prompt, self.vae.dtype, device, generator
            )

            # 5. Encode input prompt
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
            )
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            # 6. Prepare timesteps
            self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps, num_inference_steps = self.get_inverse_timesteps(num_inference_steps, inpaint_strength, device)

            # 7. Noising loop where we obtain the intermediate noised latent image for each timestep.
            num_warmup_steps = len(timesteps) - num_inference_steps * self.inverse_scheduler.order
            inverted_latents = []
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.inverse_scheduler.scale_model_input(latent_model_input, t)
                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if return_noise:
                        noise_features.append(noise_pred)
                    # regularization of the noise prediction (not in original code or paper but borrowed from Pix2PixZero)
                    if num_reg_steps > 0:
                        with torch.enable_grad():
                            for _ in range(num_reg_steps):
                                if lambda_auto_corr > 0:
                                    for _ in range(num_auto_corr_rolls):
                                        var = torch.autograd.Variable(noise_pred.detach().clone(), requires_grad=True)

                                        # Derive epsilon from model output before regularizing to IID standard normal
                                        var_epsilon = self.get_epsilon(var, latent_model_input.detach(), t)

                                        l_ac = auto_corr_loss(var_epsilon, generator=generator)
                                        l_ac.backward()

                                        grad = var.grad.detach() / num_auto_corr_rolls
                                        noise_pred = noise_pred - lambda_auto_corr * grad

                                if lambda_kl > 0:
                                    var = torch.autograd.Variable(noise_pred.detach().clone(), requires_grad=True)

                                    # Derive epsilon from model output before regularizing to IID standard normal
                                    var_epsilon = self.get_epsilon(var, latent_model_input.detach(), t)

                                    l_kld = kl_divergence(var_epsilon)
                                    l_kld.backward()

                                    grad = var.grad.detach()
                                    noise_pred = noise_pred - lambda_kl * grad

                                noise_pred = noise_pred.detach()

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.inverse_scheduler.step(noise_pred, t, latents).prev_sample
                    inverted_latents.append(latents.detach().clone())

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.inverse_scheduler.order == 0
                    ):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

            assert len(inverted_latents) == len(timesteps)
            latents = torch.stack(list(reversed(inverted_latents)), 1)

            # 8. Post-processing
            image = None
            if decode_latents:
                image = self.decode_latents(latents.flatten(0, 1))

            # 9. Convert to PIL.
            if decode_latents and output_type == "pil":
                image = self.image_processor.numpy_to_pil(image)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (latents, image)
            
            if len(noise_features) > 0:
                noise_features = torch.stack(noise_features, dim=0)
        return DiffEditInversionPipelineOutput(latents=latents, images=image, noise_features=noise_features)

