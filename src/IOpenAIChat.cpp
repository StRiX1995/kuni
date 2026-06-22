//
// Created by alex2772 on 5/9/26.
//

#include "IOpenAIChat.h"

#include <algorithm>

namespace {
AJson toOpenAIMessageJson(const IOpenAIChat::Message& message) {
    auto json = aui::to_json(message);
    auto& object = json.asObject();
    auto eraseField = [&](AStringView key) {
        std::erase_if(object, [&](const auto& entry) { return entry.first == key; });
    };

    if (message.tool_calls.empty()) {
        eraseField("tool_calls");
    } else {
        for (auto& toolCall : json["tool_calls"].asArray()) {
            auto& toolCallObject = toolCall.asObject();
            std::erase_if(toolCallObject, [](const auto& entry) { return entry.first == "index"; });
        }
    }
    if (message.tool_call_id.empty()) {
        eraseField("tool_call_id");
    }
    if (message.reasoning.empty()) {
        eraseField("reasoning");
    }
    if (message.reasoning_content.empty()) {
        eraseField("reasoning_content");
    }
    return json;
}
}

AJson AJsonConv<AVector<IOpenAIChat::Message>>::toJson(const AVector<IOpenAIChat::Message>& v) {
    AJson::Array result;
    for (const auto& message: v) {
        // reverse engineered from vscode copilot plugin
        if (message.content.contains("</{}>"_format(IOpenAIChat::EMBEDDING_TAG))) {
            auto content = std::string_view(message.content);
            auto append = [&](const IOpenAIChat::Message& msg) {
                if (msg.content.empty()) {
                    return;
                }
                result << toOpenAIMessageJson(msg);
            };
            for (;;) {
                auto tagPos = content.find("<{}>"_format(IOpenAIChat::EMBEDDING_TAG));
                append(IOpenAIChat::Message{
                    .role = message.role,
                    .content = content.substr(0, tagPos),
                });
                if (tagPos == std::string::npos) {
                    break;
                }
                content = content.substr(tagPos);
                content = content.substr(content.find(">") + 1);
                auto body = content.substr(0, content.find("</{}>"_format(IOpenAIChat::EMBEDDING_TAG)));
                append(IOpenAIChat::Message{
                    .role = message.role,
                    .content = "<attachments>",
                });
                result << AJson::Object{
                    {"role", aui::to_json(message.role)},
                    {"content",
                     AJson::Array{
                         AJson::Object{
                             {"type", "image_url"},
                             {"image_url", AJson::Object{{"url", body}}},
                         },
                     }},
                };
                append(IOpenAIChat::Message{
                    .role = message.role,
                    .content = "</attachments>",
                });
                content = content.substr(body.length());
                content = content.substr(content.find(">") + 1);
            }
            continue;
        }
        result << toOpenAIMessageJson(message);
    }
    return result;
}

void AJsonConv<IOpenAIChat::Response::Usage, void>::fromJson(const AJson& v, IOpenAIChat::Response::Usage& dst) {
    aui::zero(dst);
    dst.prompt_tokens = v["prompt_tokens"].asLongIntOpt().valueOr(0);
    dst.completion_tokens = v["completion_tokens"].asLongIntOpt().valueOr(0);
    dst.total_tokens = v["total_tokens"].asLongIntOpt().valueOr(0);
    dst.prompt_cache_hit_tokens = v["prompt_cache_hit_tokens"].asLongIntOpt()
        .valueOr([&]() -> int64_t {
            auto promptTokenDetails = v["prompt_tokens_details"].asObjectOpt();
            if (!promptTokenDetails) {
                return -1;
            }
            return (*promptTokenDetails)["cached_tokens"].asLongIntOpt().valueOr(-1); // openrouter
        });
    if (dst.prompt_cache_hit_tokens < 0) {
        dst.prompt_cache_hit_tokens = 0;
    } else {
        dst.prompt_cache_miss_tokens = v["prompt_cache_miss_tokens"].asLongIntOpt() // deepseek
            .valueOr(dst.prompt_tokens - dst.prompt_cache_hit_tokens); // openrouter
    }
}
