#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace glfs {

class JsonValue {
public:
    using object_t = std::map<std::string, JsonValue>;
    using array_t = std::vector<JsonValue>;
    using storage_t = std::variant<std::nullptr_t, bool, double, std::string, object_t, array_t>;

    JsonValue();
    JsonValue(std::nullptr_t);
    JsonValue(bool value);
    JsonValue(double value);
    JsonValue(std::string value);
    JsonValue(object_t value);
    JsonValue(array_t value);

    [[nodiscard]] bool is_null() const;
    [[nodiscard]] bool is_bool() const;
    [[nodiscard]] bool is_number() const;
    [[nodiscard]] bool is_string() const;
    [[nodiscard]] bool is_object() const;
    [[nodiscard]] bool is_array() const;

    [[nodiscard]] bool as_bool() const;
    [[nodiscard]] double as_number() const;
    [[nodiscard]] const std::string& as_string() const;
    [[nodiscard]] const object_t& as_object() const;
    [[nodiscard]] const array_t& as_array() const;

    [[nodiscard]] bool contains(const std::string& key) const;
    [[nodiscard]] const JsonValue& at(const std::string& key) const;
    [[nodiscard]] const JsonValue& at(std::size_t index) const;
    [[nodiscard]] std::size_t size() const;

    [[nodiscard]] const storage_t& storage() const;

private:
    storage_t value_;
};

JsonValue parse_json(const std::string& input);

}  // namespace glfs
