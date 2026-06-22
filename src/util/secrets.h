#pragma once
#include <toml11/types.hpp>

#include "Endpoint.h"

namespace util {

toml::basic_value<toml::type_config> secrets();

EndpointAndModel openAICompatibleEndpoint(EndpointAndModel fallback);

}
