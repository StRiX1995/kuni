#pragma once
#include "IStableDiffusionClient.h"

/**
 * @brief Stable Diffusion client supporting ComfyUI workflow API and the legacy WebUI API.
 */
struct StableDiffusionClientImpl: IStableDiffusionClient {
    Endpoint endpoint = config::ENDPOINT_SD;

    AFuture<Txt2ImgResponse> txt2img(const Txt2ImgRequest& request) override;
    AFuture<> unloadCheckpoint() override;
};
