import dotenv from "dotenv";
import open from "open";
dotenv.config();
export async function generateImageWithReplicate(prompt, options = {}) {
    const apiToken = process.env.REPLICATE_API_TOKEN;
    if (!apiToken) {
        throw new Error("REPLICATE_API_TOKEN environment variable is not set");
    }
    // Default parameters
    const defaultInput = {
        width: 768,
        height: 768,
        prompt: prompt,
        refine: "expert_ensemble_refiner",
        scheduler: "K_EULER",
        lora_scale: 0.6,
        num_outputs: 1,
        guidance_scale: 7.5,
        apply_watermark: false,
        high_noise_frac: 0.8,
        negative_prompt: "",
        prompt_strength: 0.8,
        num_inference_steps: 25,
        ...options,
    };
    const payload = {
        version: "7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        input: defaultInput,
    };
    try {
        const response = await fetch("https://api.replicate.com/v1/predictions", {
            method: "POST",
            headers: {
                Authorization: `Bearer ${apiToken}`,
                "Content-Type": "application/json",
                Prefer: "wait", // Synchronous mode
            },
            body: JSON.stringify(payload),
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || "Failed to create prediction");
        }
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }
        if (!data.output ||
            !Array.isArray(data.output) ||
            data.output.length === 0) {
            throw new Error("No output generated or invalid output format");
        }
        const imageUrl = data.output[0];
        return imageUrl;
    }
    catch (error) {
        console.error("Error generating image with Replicate:", error);
        throw error;
    }
}
//function to display the genetated image
export async function displayGeneratedImage(imageUrl) {
    await open(imageUrl);
    console.log(`Image displayed at ${imageUrl}`);
}
