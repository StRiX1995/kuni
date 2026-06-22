//
// Created by alex2772 on 2/27/26.
//

#include "OpenAITools.h"

#include <range/v3/algorithm/remove_if.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>

AJSON_FIELDS(OpenAITools::Tool::Parameters::Property, AJSON_FIELDS_ENTRY(type) AJSON_FIELDS_ENTRY(description))

AJSON_FIELDS(OpenAITools::Tool::Parameters,
             AJSON_FIELDS_ENTRY(type) AJSON_FIELDS_ENTRY(properties) AJSON_FIELDS_ENTRY(required)
                     AJSON_FIELDS_ENTRY(additionalProperties))

AJSON_FIELDS(OpenAITools::Tool, AJSON_FIELDS_ENTRY(type) AJSON_FIELDS_ENTRY(name) AJSON_FIELDS_ENTRY(description)
                                        AJSON_FIELDS_ENTRY(parameters) AJSON_FIELDS_ENTRY(strict))


OpenAITools::OpenAITools(std::initializer_list<Tool> tools) {
    ALOG_TRACE("OpenAITools") << "OpenAITools";
    for (const auto& tool : tools) {
        AUI_ASSERT(tool.handler != nullptr);
        mHandlers[tool.name] = tool;
    }
}

static AString removeControlCharacters(AString input) {
    // toml11 library inserts beautiful formatting to the exceptions, and JSON is not happy about it.
    input.bytes().erase(ranges::remove_if(input.bytes(), [](char c) {
        switch (c) {
            case '\n':
            case '\t':
                return false;
            default:
            return 0x00 <= c && c < 0x20;
        }
    }), input.bytes().end());
    return input;
}

AFuture<AVector<IOpenAIChat::Message>> OpenAITools::handleToolCalls(const AVector<IOpenAIChat::Message::ToolCall>& toolCalls,
    const _<MetricsBreadcumbs>& metricsBreadCumbs) {
    ALOG_TRACE("OpenAITools") << "handleToolCalls";
    mHadSuccessfulUserVisibleAction = false;

    AVector<IOpenAIChat::Message> result(toolCalls.size());
    // First resolve retrieval, generation and other private work. User-visible actions are
    // delivered only after all regular tools in this assistant response have completed.
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t index = 0; index < toolCalls.size(); ++index) {
            const auto& toolCall = toolCalls[index];
            auto handler = mHandlers.contains(toolCall.function.name);
            const bool deferred = handler && handler->second.deferExecution;
            if ((pass == 0 && deferred) || (pass == 1 && !deferred)) {
                continue;
            }

            AString handlerResult;
            bool successfulUserVisibleAction = false;
            try {
                if (handler) {
                    // A handler is allowed to replace ctx.tools completely (open_chat_by_id does this).
                    // Keep a local copy so the coroutine closure and runtime policy remain alive even if
                    // mHandlers is mutated while the handler is suspended.
                    auto tool = handler->second;
                    AOptional<MetricsBreadcumbs::Point> point;
                    if (metricsBreadCumbs) {
                        point.emplace(metricsBreadCumbs, "function", toolCall.function.name);
                    }
                    handlerResult = co_await tool.handler({
                        .tools = *this,
                        .args = AJson::fromString(toolCall.function.arguments),
                        .allToolCalls = toolCalls,
                    });
                    if (onAfterToolCall) {
                        onAfterToolCall(toolCall.function.name);
                    }
                    successfulUserVisibleAction = tool.userVisible &&
                        !handlerResult.trim().lowercase().startsWith("error");
                } else {
                    handlerResult = "tool \"" + toolCall.function.name +
                        "\" is not currently available. Please use another tool instead.";
                }
            } catch (const AException& e) {
                ALogger::err("OpenAITools") << "error while executing \"{}\" tool: "_format(toolCall.function.name) << e;
                handlerResult = "error while executing \"{}\" tool: {}"_format(toolCall.function.name, e.getMessage());
            }

            result[index] = IOpenAIChat::Message{
                .role = IOpenAIChat::Message::Role::TOOL,
                .content = removeControlCharacters(std::move(handlerResult)),
                .tool_call_id = toolCall.id,
            };
            if (successfulUserVisibleAction) {
                mHadSuccessfulUserVisibleAction = true;
            }
        }
    }
    co_return result;

}

AJson OpenAITools::asJson() const {
    ALOG_TRACE("OpenAITools") << "asJson";
    return ranges::view::transform(mHandlers, [](const auto& tool) {
        return AJson::Object{
            {"type", tool.second.type },
            {"function", aui::to_json(tool.second) },
        };
    }) | ranges::to<AJson::Array>();
}
