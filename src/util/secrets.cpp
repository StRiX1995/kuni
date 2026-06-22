#include "secrets.h"

#include "AUI/IO/APath.h"
#include "AUI/Logging/ALogger.h"

#include <toml.hpp>

static constexpr auto LOG_TAG = "secrets";
static constexpr auto TEMPLATE = R"(
# SECRETS file, Kuni project, https://github.com/alex2772/Kuni/
# WARNING - SENSITIVE DATA! DO NOT SHARE NOR COMMIT THIS FILE!!!

[telegram_api]
# tdlib API key, see
# https://core.telegram.org/api/obtaining_api_id
id = 0
hash = ""

[telegram_proxy]
# Optional MTProxy used by TDLib. Values can be copied from a
# tg://proxy?server=...&port=...&secret=... link.
enabled = false
server = ""
port = 443
secret = ""

[openai_compatible]
# Optional external OpenAI-compatible chat provider.
enabled = false
base_url = "https://example.com/v1/"
model = ""
auth_key = ""

[comfyui]
# Optional ComfyUI API-format workflow adapter.
enabled = false
base_url = "http://localhost:7860/"
workflow_path = ""
positive_node = "39:0"
negative_node = "39:1"
sampler_node = "11"
size_switch_node = "261"
output_node = "248"

[ollama]
# uncomment and specify this if you want Ollama web search
# bearer_key = ""

[elevenlabs]
# ElevenLabs API key for TTS
# api_key = ""
# Optional default voice ID. If omitted, the built-in default voice id is used.
# voice_id = "pPdl9cQBQq4p6mRkZy2Z"
)";


toml::basic_value<toml::type_config> util::secrets() {
    static auto data = [] {
        const auto location = APath("data") / "secrets.toml";
        if (!location.isRegularFileExists()) {
            location.parent().makeDirs();
            AFileOutputStream(location) << TEMPLATE;
            ALogger::err(LOG_TAG) << R"(
########################################################################################################################
#                                                      IMPORTANT                                                       #
#                                         Please populate data/secrets.toml                                            #
#                                                     and restart                                                      #
########################################################################################################################
)";
            std::exit(-1);
        }

        try {
            toml::color::enable();
            auto data = toml::parse(location, toml::spec::v(1,1,0));;

            {
                const auto& value = data["telegram_api"]["id"];
                if (value.as_integer() == 0) {
                    ALogger::err(LOG_TAG) << toml::format_error((toml::make_error_info("telegram_api.id should be populated", value, "the actual value is 0")));
                    std::exit(-1);
                }
            }
            {
                const auto& value = data["telegram_api"]["hash"];
                if (value.as_string().empty()) {
                    ALogger::err(LOG_TAG) << toml::format_error((toml::make_error_info("telegram_api.hash should be populated", value, "the actual string is empty")));
                    std::exit(-1);
                }
            }
            return data;
        } catch (const std::exception& e) {
            ALogger::err(LOG_TAG) << "Failed to parse " << location;
            throw;
        }
    }();
    return data;
}

EndpointAndModel util::openAICompatibleEndpoint(EndpointAndModel fallback) {
    auto data = secrets();
    if (!data.contains("openai_compatible")) {
        return fallback;
    }

    auto& config = data["openai_compatible"];
    if (!config.contains("enabled") || !config["enabled"].as_boolean()) {
        return fallback;
    }

    const auto baseUrl = config["base_url"].as_string();
    const auto model = config["model"].as_string();
    const auto authKey = config["auth_key"].as_string();
    if (baseUrl.empty() || model.empty()) {
        throw AException("[openai_compatible].base_url and model must be populated in data/secrets.toml");
    }

    auto normalizedBaseUrl = AString(baseUrl);
    if (!normalizedBaseUrl.endsWith('/')) {
        normalizedBaseUrl += '/';
    }
    return {
        .endpoint = {
            .baseUrl = std::move(normalizedBaseUrl),
            .bearerKey = authKey,
        },
        .model = model,
    };
}
