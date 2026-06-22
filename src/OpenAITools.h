#pragma once
#include "AUI/Common/AString.h"
#include "AUI/Common/AVector.h"
#include "AUI/Json/AJson.h"
#include "IOpenAIChat.h"
#include "MetricsBreadcumbs.h"

struct OpenAITools {
    struct Ctx {
        OpenAITools& tools;
        AJson args;
        const AVector<IOpenAIChat::Message::ToolCall>& allToolCalls;
    };
    using Handler = std::function<AFuture<AString>(Ctx ctx)>;

    struct Tool {
        AString type = "function";
        AString name;
        AString description;
        struct Parameters {
            AString type = "object";
            struct Property {
                AString type = "string";
                AString description;
            };
            AMap<AString, Property> properties;
            AVector<AString> required; // required properties
            bool additionalProperties = false;
        } parameters;
        // Most Kuni tools have genuinely optional or mutually exclusive fields.
        // OpenAI strict mode requires every property to be listed in required,
        // so these schemas must use regular (non-strict) function calling.
        bool strict = false;
        // Deferred tools are executed after every regular tool from the same assistant response.
        // These fields are runtime policy and are intentionally not serialized into the tool schema.
        bool deferExecution = false;
        bool userVisible = false;
        Handler handler;
    };

    OpenAITools(std::initializer_list<Tool> tools);

    /**
     * @brief Optional hook fired after each tool call handler completes successfully.
     * Not called if the handler throws. Set by AppBase::updateTools to emit AppBase::toolCallFired.
     */
    std::function<void(const AString& toolName)> onAfterToolCall;

    AFuture<AVector<IOpenAIChat::Message>> handleToolCalls(
        const AVector<IOpenAIChat::Message::ToolCall>& toolCalls,
        const _<MetricsBreadcumbs>& metricsBreadCumbs = nullptr);

    [[nodiscard]] bool hadSuccessfulUserVisibleAction() const noexcept {
        return mHadSuccessfulUserVisibleAction;
    }

    AJson asJson() const;

    [[nodiscard]] AMap<AString, Tool> handlers() const { return mHandlers; }

    void insert(Tool tool) { mHandlers[tool.name] = std::move(tool); }

private:
    AMap<AString, Tool> mHandlers;
    bool mHadSuccessfulUserVisibleAction = false;
};
