#include "ImageGenerator.h"

#include <random>
#include <range/v3/algorithm/count_if.hpp>

#include "KuniCharacter.h"
#include "AUI/Logging/ALogger.h"
#include "AUI/Util/kAUI.h"
#include <range/v3/view/transform.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/reverse.hpp>

#include "IOpenAIChat.h"
#include "OpenAIChatImpl.h"
#include "AUI/Image/png/PngImageLoader.h"
#include "AUI/IO/AFileInputStream.h"

static constexpr auto LOG_TAG = "ImageGenerator";
static constexpr auto TRIAL_COUNT = 10;

static AJson parseResponse(AString content) {
    // Basic JSON extraction if the model wrapped it in markdown
    if (content.contains("```json")) {
        content = content.split("```json").at(1).split("```").at(0);
    } else if (content.contains("```")) {
        content = content.split("```").at(1).split("```").at(0);
    }
    return AJson::fromString(content);
}

AFuture<ImageGenerator::GalleryImage> ImageGenerator::generate(AString description) {
    ALOG_TRACE(LOG_TAG) << "generate: " << description;
    AString appearancePrompt = Kuni_character::getAppearancePrompt();
    int trialIndex = 0;

    naxyi:
    ALogger::info(LOG_TAG) << "Engineering initial prompt for: " << description;
    PromptPair currentPrompt {
        .positive = "",
        .negative = "(text:2), (signature:2), raw photo",
    };
    co_await engineerPrompt(currentPrompt, description, appearancePrompt);
    AString firstFeedback;

    AString descriptionWithAppearance = "<character name=\"Kuni\" canonical_description>\n{}\n</character canonical_description>\n<user_description overrides_canonical=\"true\">\n{}\n</user_description overrides_canonical=\"true\">"_format(appearancePrompt, description);

    while (trialIndex <= TRIAL_COUNT) {
        try {
            ++trialIndex;
            static std::default_random_engine ge(std::time(nullptr));
            {
                ALogger::info(LOG_TAG) << "Iteration " << trialIndex << " with prompt:\npositive=" << currentPrompt.positive << "\n\nnegative=" << currentPrompt.negative;

                IStableDiffusionClient::Txt2ImgResponse response;
                try
                {
                    response = co_await mSdClient->txt2img({
                        .prompt = currentPrompt.positive,
                        .negative_prompt = currentPrompt.negative,
                        .steps =  30,
                        .cfg_scale = std::uniform_real_distribution<>(1.0, 5.0)(ge),
                        .width = std::uniform_int_distribution<>(768, 1400)(ge),
                        .height = std::uniform_int_distribution<>(768, 1400)(ge),
                        .enable_hr = true,
                        .hr_scale = 1.5,
                        .hr_upscaler = "Latent",
                        .hr_second_pass_steps = 10,
                        .denoising_strength = 0.7,
                    });
                } catch (const AException& e) {
                    ALogger::err(LOG_TAG) << "Stable diffusion failed:: " << e;
                    goto tryGallery;
                }
                if (response.images.empty()) {
                    throw AException("Stable Diffusion returned no images");
                }
                auto lastImage = response.images[0];
                PngImageLoader::save(AFileOutputStream{ "image_generator_tmp.png" }, *lastImage);
                //Unload SD checkpoint after generation
                try {
                    co_await mSdClient->unloadCheckpoint();
                    ALogger::info(LOG_TAG) << "Checkpoint unloaded from VRAM";
                } catch (const AException& e) {
                    ALogger::warn(LOG_TAG) << "Failed to unload checkpoint: " << e;
                }

                ALogger::info(LOG_TAG) << "Assessing image...";
                auto assessment = co_await assessImage(*lastImage, descriptionWithAppearance);

                if (assessment.satisfied) {
                    ALogger::info(LOG_TAG) << "Satisfied with the result. " << assessment.feedback;
                    const auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::system_clock::now().time_since_epoch())
                                               .count();
                    auto dst = APath("data/gallery/{}.png"_format(timestamp));
                    dst.parent().makeDirs();
                    PngImageLoader::save(AFileOutputStream{ dst }, *lastImage);
                    co_return GalleryImage{ .image = lastImage, .path = dst.absolute() };
                }
                if (firstFeedback.empty()) {
                    firstFeedback = assessment.feedback;
                }

                ALogger::info(LOG_TAG) << "Not satisfied. Feedback: " << assessment.feedback;
                co_await engineerPrompt(currentPrompt, description, appearancePrompt, assessment.feedback);
            }


            tryGallery:
            // // in case SD fails, let's try a photo from gallery.
            // auto galleryFiles = APath("data/gallery").listDir(AFileListFlags::REGULAR_FILES);
            // if (galleryFiles.empty())
            // {
            //     continue;
            // }
            // auto randomFile = galleryFiles[std::uniform_int_distribution<>(0, galleryFiles.size() - 1)(ge)];
            // ALogger::info(LOG_TAG) << "Trying to supply image from gallery: " << randomFile;
            // auto lastImage = AImage::fromBuffer(AByteBuffer::fromStream(AFileInputStream{ randomFile }));
            // auto assessment = co_await assessImage(*lastImage, descriptionWithAppearance);
            // if (assessment.satisfied) {
            //     ALogger::info(LOG_TAG) << "Satisfied with the image from gallery: " << assessment.feedback;
            //     co_return lastImage;
            // }

            if (trialIndex % size_t(std::sqrt(TRIAL_COUNT)) == 0) {
                ALogger::info(LOG_TAG) << "Last trial failed. Retrying with different prompt...";
                goto naxyi;
            }
        } catch (const AException& e) {
            ALogger::err(LOG_TAG) << "Failed to generate image: " << e;
        }
    }

    throw AException("can't find image: feedback: \"{}\"; make photo_desc shorter"_format(firstFeedback));
}

static constexpr auto LOL_WHAT = {
    "explicit nudity",
    "explicit nude",
    "explicit erotic",
    "pussy",
    "breasts",
    "nsfw",
    "genital",
    "vagina",
    "penis",
};

AFuture<> ImageGenerator::engineerPrompt(PromptPair& out, const AString& description, const AString& appearancePrompt, const AString& feedback) {
    ALOG_TRACE(LOG_TAG) << "engineerPrompt description=" << description << " appearancePrompt=" << appearancePrompt << " feedback=" << feedback;
    auto safeDescription = description;
    for (const auto& word : LOL_WHAT) {
        safeDescription.replaceAll(word, "");
    }
    auto params = mChatParams;
    params.systemPrompt = R"(
You are an expert Stable Diffusion prompt engineer.
Your task is to transform a freeform description into a high-quality, descriptive Stable Diffusion prompt.
You must also integrate the provided character appearance details.

Guidelines:
- Use descriptive keywords, artist names, and technical terms (e.g., "hyperrealistic", "8k", "masterpiece").
- Ensure the character's appearance matches the provided appearance prompt. Appearance prompt includes both freeform
  description and stable-diffusion-optimized prompt. Base your prompt on the character's stable-diffusion prompt,
  preserving original aesthetics of the character, avoid altering original character design.
- Who made the image and how? Almost always it would be selfie, unless description explicitly specifies a photographer.
- If previous prompt iteration were provided, adjust them according to feedback and make sure it still satisfies
  original character design and desired photo description.

# Characters

<character name="Kuni">
{}
</character name="Kuni">

# Desired photo description

```
{}
```

# Output formatting

Respond in JSON object format with the following fields:

- "positivePrompt": string, positive prompt
- "negativePrompt": string, negative prompt

)"_format(appearancePrompt, safeDescription);

    auto messages = [&] {
        AString message;

        message += "# Previous prompt iteration \n";
        message += "Positive prompt: ";
        message += out.positive;
        message += "\n\n";
        message += "Negative prompt: ";
        message += out.negative;
        message += "\n\n";

        if (!feedback.empty()) {
            message += "# Feedback to previous prompt iteration\n";
            message += feedback;
        }

        message += R"(

<instructions>
When improving the prompt:
- Prefer Stable Diffusion weighting syntax like (term:1.2) or (phrase:1.5).
- Do not make the prompt longer just to improve emphasis.
- If the prompt is too long, shorten it by removing filler words.
- Only add new words if the image is missing a critical concept.
- Keep the final prompt short, structured, and friendly for Stable Diffusion.
- Do not alter original character design
- "positivePrompt": string, a slightly modified version of the current positive prompt to fix the issues.
- "negativePrompt": string, a slightly modified version of the current negative prompt to fix the issues.

Positive prompt is what to include to the image.

Negative prompt is what to avoid in the image.
</instructions>
    )";

        message += "\nGenerate SD prompt:";
        return AVector<IOpenAIChat::Message>{
            IOpenAIChat::Message{
                .role = IOpenAIChat::Message::Role::USER,
                .content = message,
            }
        };
    }();
    auto response = co_await mOpenAI->chat(params, messages);
    naxyi:
    if (response.choices.empty()) {
        throw AException("OpenAI returned no choices for initial prompt engineering");
    }
    auto content = response.choices[0].message.content;
    auto json = parseResponse(content);
    out = {
        .positive = json["positivePrompt"].asString(),
        .negative = json["negativePrompt"].asString(),
    };

    for (const auto&[name, prompt] : std::array {std::make_pair("positive", &out.positive), std::make_pair("negative", &out.negative) }) {
        prompt->replaceAll(") ", "), "); // add commas
        auto wordCount = ranges::count_if(*prompt, [](char c ){ return c == ' '; });
        if (wordCount > 60) {
            // long prompts to stable diffusion are generally distorting the character base design.
            if (messages.size() > 3) {
                throw AException("adjusted {} prompt is too long."_format(name));
            }
            messages << IOpenAIChat::Message{
                .role = IOpenAIChat::Message::Role::USER,
                .content = "Adjusted {} prompt is too long. Shorten it to 50 words or less.; restructure or adjust word (weights:1.5) instead"_format(name)
            };
            response = co_await mOpenAI->chat(params, messages);
            goto naxyi;
        }
    }

    for (const auto& badWord : LOL_WHAT) {
        if (description.contains(badWord)) {
            out.positive += " ";
            out.positive += badWord;
        }
    }


    co_return;
}

AFuture<ImageGenerator::AssessmentResult> ImageGenerator::assessImage(const AImage& image, const AString& description) {
    ALOG_TRACE(LOG_TAG) << "assessImage description=" << description;
    auto params = mChatParams;
    // Note: mChatParams.config should ideally be a vision-capable model.
    params.systemPrompt = R"(
 You are a practical image critic and Stable Diffusion quality gate.

You will be shown an image generated from a user description and a character appearance prompt.
 Your job is to decide whether the image is good enough to show to the user.
 Judge the overall result and reject only material, clearly visible problems. Do not demand pixel-perfect or literal
 reproduction of every prompt detail.

 Reject the image if ANY of the following are clearly true:
- There is a major anatomical defect, such as a duplicated face or limb, fused body parts, or a badly broken face.
- The image is unreadable or has severe blur, corruption, melting, or other prominent generation artifacts.
- The number of subjects is wrong, or the main requested subject/action/scene is substantially absent.
- The character's core identity is missing or replaced by a visibly different character.
- The result is clearly unusable or unpleasant rather than merely imperfect.
- If the user's description overrides a canonical detail, the image must follow the description, not the canonical
  detail. Canonical design is the fallback only when the description does not specify an alternative.
- "explicit nudity", unless asked.

Do NOT reject solely because of:
- Minor variations in hair length, clothing ornament, lighting, background objects, or other secondary details.
- Details that are subtle, occluded, cropped out, or irrelevant to the requested composition.
- A plausible stylistic interpretation of the prompt.
- Small imperfections that an ordinary viewer would not notice without close inspection.
- Uncertainty about a minor detail.

 Important rule:
- Set "satisfied" to true when the image is visually appealing, has no major visible defect, preserves the character's
  recognizable identity, and broadly fulfills the user's request.
- Use canonical character design as the default baseline, but let the user's description override any conflicting details.
- If canonical says one thing and description says another, judge the image against the description.
- Treat the canonical description as identity guidance, not as a checklist requiring every known fact to be visible.

Output your assessment in JSON format with the following fields:
- "satisfied": boolean, true if the image is high quality and matches the description.
- "feedback": string, explaining what's wrong if not satisfied.

{}
)";
    params.systemPrompt = params.systemPrompt.format(description);

    AVector<IOpenAIChat::Message> messages = {
        IOpenAIChat::Message{
            .role = IOpenAIChat::Message::Role::USER,
            .content = "Assess this image: " + IOpenAIChat::embedImage(image)
        }
    };
    auto response = co_await mOpenAI->chat(params, messages);

    if (response.choices.empty()) {
        throw AException("OpenAI returned no choices for image assessment");
    }
    auto responseContent = response.choices[0].message.content;

    try {
        auto json = parseResponse(responseContent);
        AssessmentResult result{
            .satisfied = json["satisfied"].asBool(),
            .feedback = json["feedback"].asString(),
        };
        co_return result;
    } catch (const AException& e) {
        ALogger::err(LOG_TAG) << "Failed to parse assessment JSON: " << e << "\nContent: " << responseContent;
        // Fallback: assume satisfied if parsing fails to avoid infinite loops, but log error
        co_return AssessmentResult{.satisfied = false, .feedback = "" };
    }
}
