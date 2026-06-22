#include "StableDiffusionClientImpl.h"
#include "AUI/Curl/ACurl.h"
#include "AUI/IO/AFileInputStream.h"
#include "AUI/Json/Conversion.h"
#include "AUI/Logging/ALogger.h"
#include "AUI/Thread/AThread.h"
#include <range/v3/view/transform.hpp>
#include <range/v3/range/conversion.hpp>

#include <chrono>
#include <limits>
#include <optional>
#include <random>

#include "config.h"
#include "util/secrets.h"

static constexpr auto LOG_TAG = "StableDiffusionClient";
using namespace std::chrono_literals;

namespace {
struct ComfyUIConfig {
    AString baseUrl;
    APath workflowPath;
    AString positiveNode;
    AString negativeNode;
    AString samplerNode;
    AString sizeSwitchNode;
    AString outputNode;
};

std::optional<ComfyUIConfig> comfyUIConfig() {
    auto secrets = util::secrets();
    if (!secrets.contains("comfyui")) {
        return std::nullopt;
    }
    auto& config = secrets["comfyui"];
    if (!config.contains("enabled") || !config["enabled"].as_boolean()) {
        return std::nullopt;
    }
    auto value = [&](const char* key) -> AString {
        if (!config.contains(key) || !config[key].is_string() || config[key].as_string().empty()) {
            throw AException("[comfyui].{} must be populated in data/secrets.toml"_format(key));
        }
        return config[key].as_string();
    };
    auto baseUrl = value("base_url");
    if (!baseUrl.endsWith('/')) {
        baseUrl += '/';
    }
    return ComfyUIConfig {
        .baseUrl = std::move(baseUrl),
        .workflowPath = value("workflow_path"),
        .positiveNode = value("positive_node"),
        .negativeNode = value("negative_node"),
        .samplerNode = value("sampler_node"),
        .sizeSwitchNode = value("size_switch_node"),
        .outputNode = value("output_node"),
    };
}

AJson& workflowInputs(AJson& workflow, const AString& nodeId) {
    if (!workflow.contains(nodeId) || !workflow[nodeId].isObject() || !workflow[nodeId].contains("inputs")) {
        throw AException("ComfyUI workflow node {} is missing or has no inputs"_format(nodeId));
    }
    return workflow[nodeId]["inputs"];
}

int64_t randomSeed() {
    static std::mt19937_64 engine(std::random_device{}());
    return std::uniform_int_distribution<int64_t>(0, std::numeric_limits<int64_t>::max())(engine);
}

AFuture<IStableDiffusionClient::Txt2ImgResponse> comfyTxt2Img(
    const ComfyUIConfig& config,
    const IStableDiffusionClient::Txt2ImgRequest& request) {
    auto workflow = AJson::fromBuffer(AByteBuffer::fromStream(AFileInputStream(config.workflowPath)));
    if (!workflow.isObject()) {
        throw AException("ComfyUI workflow must be an API-format JSON object");
    }

    workflowInputs(workflow, config.positiveNode)["text"] = request.prompt;
    workflowInputs(workflow, config.negativeNode)["text"] = request.negative_prompt;

    auto& sampler = workflowInputs(workflow, config.samplerNode);
    const auto seed = request.seed < 0 ? randomSeed() : request.seed;
    sampler["seed"] = seed;
    sampler["steps"] = request.steps;
    sampler["cfg"] = request.cfg_scale;

    // The supplied workflow has four preconfigured SDXL sizes. Preserve those sizes and select by aspect ratio.
    const auto aspectRatio = static_cast<double>(request.width) / static_cast<double>(request.height);
    workflowInputs(workflow, config.sizeSwitchNode)["select"] = aspectRatio > 1.2 ? 1 : aspectRatio < 0.85 ? 2 : 4;

    const auto submitResponse = AJson::fromBuffer(
        (co_await ACurl::Builder(config.baseUrl + "prompt")
             .withMethod(ACurl::Method::HTTP_POST)
             .withHeaders({"Content-Type: application/json"})
             .withBody(AJson::toString(AJson::Object{{"prompt", std::move(workflow)}}).toStdString())
             .withTimeout(30s)
             .runAsync())
            .body);
    if (!submitResponse.isObject() || !submitResponse.contains("prompt_id") || !submitResponse["prompt_id"].isString()) {
        throw AException("ComfyUI rejected workflow: {}"_format(AJson::toString(submitResponse)));
    }
    const auto promptId = submitResponse["prompt_id"].asString();
    ALogger::info(LOG_TAG) << "ComfyUI prompt queued: " << promptId;

    const auto deadline = std::chrono::steady_clock::now() + config::REQUEST_TIMEOUT;
    for (;;) {
        if (std::chrono::steady_clock::now() >= deadline) {
            throw AException("ComfyUI generation timed out: {}"_format(promptId));
        }
        co_await AThread::asyncSleep(500ms);
        const auto history = AJson::fromBuffer(
            (co_await ACurl::Builder(config.baseUrl + "history/" + promptId).withTimeout(30s).runAsync()).body);
        if (!history.isObject() || !history.contains(promptId)) {
            continue;
        }

        const auto& job = history[promptId];
        if (!job.isObject() || !job.contains("outputs") || !job["outputs"].isObject()) {
            if (job.contains("status") && job["status"].isObject() &&
                job["status"].contains("completed") && job["status"]["completed"].asBool()) {
                throw AException("ComfyUI completed without outputs: {}"_format(AJson::toString(job["status"])));
            }
            continue;
        }
        const auto& outputs = job["outputs"];
        if (!outputs.contains(config.outputNode) || !outputs[config.outputNode].contains("images")) {
            throw AException("ComfyUI output node {} produced no images"_format(config.outputNode));
        }

        IStableDiffusionClient::Txt2ImgResponse result;
        for (const auto& image : outputs[config.outputNode]["images"].asArray()) {
            auto imageResponse = co_await ACurl::Builder(config.baseUrl + "view")
                                     .withParams({
                                         {"filename", image["filename"].asString()},
                                         {"subfolder", image["subfolder"].asString()},
                                         {"type", image["type"].asString()},
                                     })
                                     .withTimeout(30s)
                                     .runAsync();
            result.images << AImage::fromBuffer(imageResponse.body);
        }
        result.info = job;
        co_return result;
    }
}
}

AJSON_FIELDS(IStableDiffusionClient::Txt2ImgRequest,
             AJSON_FIELDS_ENTRY(prompt)
             AJSON_FIELDS_ENTRY(negative_prompt)
             AJSON_FIELDS_ENTRY(styles)
             AJSON_FIELDS_ENTRY(seed)
             AJSON_FIELDS_ENTRY(subseed)
             AJSON_FIELDS_ENTRY(subseed_strength)
             AJSON_FIELDS_ENTRY(seed_resize_from_h)
             AJSON_FIELDS_ENTRY(seed_resize_from_w)
             AJSON_FIELDS_ENTRY(sampler_name)
             AJSON_FIELDS_ENTRY(scheduler)
             AJSON_FIELDS_ENTRY(batch_size)
             AJSON_FIELDS_ENTRY(n_iter)
             AJSON_FIELDS_ENTRY(steps)
             AJSON_FIELDS_ENTRY(cfg_scale)
             AJSON_FIELDS_ENTRY(width)
             AJSON_FIELDS_ENTRY(height)
             AJSON_FIELDS_ENTRY(send_images)
             AJSON_FIELDS_ENTRY(override_settings)
             AJSON_FIELDS_ENTRY(save_images)
             AJSON_FIELDS_ENTRY(enable_hr)
             AJSON_FIELDS_ENTRY(hr_scale)
             AJSON_FIELDS_ENTRY(hr_upscaler)
             AJSON_FIELDS_ENTRY(hr_second_pass_steps)
             AJSON_FIELDS_ENTRY(denoising_strength))

AFuture<IStableDiffusionClient::Txt2ImgResponse> StableDiffusionClientImpl::txt2img(const Txt2ImgRequest& request) {
    ALOG_TRACE(LOG_TAG) << "txt2img";
    if (auto comfy = comfyUIConfig()) {
        co_return co_await comfyTxt2Img(*comfy, request);
    }
    auto query = AJson::toString(aui::to_json(request));
    ALOG_TRACE(LOG_TAG) << "Query: " << query;

    AVector<AString> headers = {"Content-Type: application/json"};
    if (!endpoint.bearerKey.empty()) {
        headers << "Authorization: Bearer {}"_format(endpoint.bearerKey);
    }

    auto responseBody = (co_await ACurl::Builder(endpoint.baseUrl + "sdapi/v1/txt2img")
                                           .withMethod(ACurl::Method::HTTP_POST)
                                           .withHeaders(std::move(headers))
                                           .withBody(query.toStdString())
                                           .withTimeout(config::REQUEST_TIMEOUT)
                                           .runAsync())
                                          .body;
    auto response = AJson::fromBuffer(responseBody);

    ALOG_TRACE(LOG_TAG) << "Response: " << AJson::toString(response);
    StableDiffusionClientImpl::Txt2ImgResponse res;
    res.images = response["images"].asArray() | ranges::views::transform([](const AJson& v) { return AImage::fromBuffer(AByteBuffer::fromBase64String(v.asString())); }) | ranges::to_vector;
    res.info = response["info"];
    co_return res;
}

AFuture<> StableDiffusionClientImpl::unloadCheckpoint() {
    ALOG_TRACE(LOG_TAG) << "unloadCheckpoint";

    if (auto comfy = comfyUIConfig()) {
        co_await ACurl::Builder(comfy->baseUrl + "free")
            .withMethod(ACurl::Method::HTTP_POST)
            .withHeaders({"Content-Type: application/json"})
            .withBody(R"({"unload_models":true,"free_memory":true})")
            .withTimeout(30s)
            .runAsync();
        co_return;
    }

    AVector<AString> headers = {"Content-Type: application/json"};
    if (!endpoint.bearerKey.empty()) {
        headers << "Authorization: Bearer {}"_format(endpoint.bearerKey);
    }

    co_await ACurl::Builder(endpoint.baseUrl + "sdapi/v1/unload-checkpoint")
        .withMethod(ACurl::Method::HTTP_POST)
        .withHeaders(std::move(headers))
        .withTimeout(config::REQUEST_TIMEOUT)
        .runAsync();

    co_return;
}
