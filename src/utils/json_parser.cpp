#include "utils/json_parser.h"

#include <cctype>
#include <charconv>
#include <cmath>
#include <limits>
#include <sstream>

namespace glfs {

namespace {

class Parser {
public:
    explicit Parser(std::string input) : input_(std::move(input)) {}

    JsonValue parse() {
        skip_ws();
        JsonValue value = parse_value();
        skip_ws();
        if (!eof()) {
            throw std::runtime_error("unexpected trailing characters in JSON");
        }
        return value;
    }

private:
    [[nodiscard]] bool eof() const { return pos_ >= input_.size(); }

    [[nodiscard]] char peek() const {
        if (eof()) {
            return '\0';
        }
        return input_[pos_];
    }

    char get() {
        if (eof()) {
            throw std::runtime_error("unexpected end of JSON input");
        }
        return input_[pos_++];
    }

    void skip_ws() {
        while (!eof() && std::isspace(static_cast<unsigned char>(peek()))) {
            ++pos_;
        }
    }

    JsonValue parse_value() {
        skip_ws();
        const char c = peek();
        if (c == '{') {
            return parse_object();
        }
        if (c == '[') {
            return parse_array();
        }
        if (c == '"') {
            return JsonValue(parse_string());
        }
        if (c == 't') {
            consume_literal("true");
            return JsonValue(true);
        }
        if (c == 'f') {
            consume_literal("false");
            return JsonValue(false);
        }
        if (c == 'n') {
            consume_literal("null");
            return JsonValue(nullptr);
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            return JsonValue(parse_number());
        }
        throw std::runtime_error("invalid JSON value");
    }

    JsonValue parse_object() {
        JsonValue::object_t object;
        expect('{');
        skip_ws();
        if (peek() == '}') {
            get();
            return JsonValue(std::move(object));
        }
        while (true) {
            skip_ws();
            if (peek() != '"') {
                throw std::runtime_error("JSON object key must be a string");
            }
            std::string key = parse_string();
            skip_ws();
            expect(':');
            skip_ws();
            object.emplace(std::move(key), parse_value());
            skip_ws();
            const char c = get();
            if (c == '}') {
                break;
            }
            if (c != ',') {
                throw std::runtime_error("expected ',' or '}' in JSON object");
            }
        }
        return JsonValue(std::move(object));
    }

    JsonValue parse_array() {
        JsonValue::array_t array;
        expect('[');
        skip_ws();
        if (peek() == ']') {
            get();
            return JsonValue(std::move(array));
        }
        while (true) {
            array.emplace_back(parse_value());
            skip_ws();
            const char c = get();
            if (c == ']') {
                break;
            }
            if (c != ',') {
                throw std::runtime_error("expected ',' or ']' in JSON array");
            }
            skip_ws();
        }
        return JsonValue(std::move(array));
    }

    std::string parse_string() {
        expect('"');
        std::string result;
        while (true) {
            if (eof()) {
                throw std::runtime_error("unterminated JSON string");
            }
            char c = get();
            if (c == '"') {
                break;
            }
            if (c == '\\') {
                if (eof()) {
                    throw std::runtime_error("unterminated escape sequence");
                }
                char esc = get();
                switch (esc) {
                    case '"': result.push_back('"'); break;
                    case '\\': result.push_back('\\'); break;
                    case '/': result.push_back('/'); break;
                    case 'b': result.push_back('\b'); break;
                    case 'f': result.push_back('\f'); break;
                    case 'n': result.push_back('\n'); break;
                    case 'r': result.push_back('\r'); break;
                    case 't': result.push_back('\t'); break;
                    case 'u': {
                        // Minimal unicode escape support: preserve as-is if decoding is unavailable.
                        if (pos_ + 4 > input_.size()) {
                            throw std::runtime_error("invalid unicode escape");
                        }
                        result.append("\\u");
                        for (int i = 0; i < 4; ++i) {
                            result.push_back(get());
                        }
                        break;
                    }
                    default:
                        throw std::runtime_error("invalid escape sequence");
                }
                continue;
            }
            result.push_back(c);
        }
        return result;
    }

    double parse_number() {
        std::size_t start = pos_;
        if (peek() == '-') {
            ++pos_;
        }
        while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
            ++pos_;
        }
        if (!eof() && peek() == '.') {
            ++pos_;
            while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
                ++pos_;
            }
        }
        if (!eof() && (peek() == 'e' || peek() == 'E')) {
            ++pos_;
            if (!eof() && (peek() == '+' || peek() == '-')) {
                ++pos_;
            }
            while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) {
                ++pos_;
            }
        }
        const std::string token = input_.substr(start, pos_ - start);
        std::size_t consumed = 0;
        double value = 0.0;
        try {
            value = std::stod(token, &consumed);
        } catch (const std::exception&) {
            throw std::runtime_error("invalid JSON number");
        }
        if (consumed != token.size()) {
            throw std::runtime_error("invalid JSON number");
        }
        if (!std::isfinite(value)) {
            throw std::runtime_error("JSON number out of range");
        }
        return value;
    }

    void consume_literal(const char* literal) {
        for (const char* p = literal; *p != '\0'; ++p) {
            if (get() != *p) {
                throw std::runtime_error("invalid JSON literal");
            }
        }
    }

    void expect(char c) {
        if (get() != c) {
            throw std::runtime_error(std::string("expected '") + c + "'");
        }
    }

    std::string input_;
    std::size_t pos_ = 0;
};

}  // namespace

JsonValue::JsonValue() : value_(nullptr) {}
JsonValue::JsonValue(std::nullptr_t) : value_(nullptr) {}
JsonValue::JsonValue(bool value) : value_(value) {}
JsonValue::JsonValue(double value) : value_(value) {}
JsonValue::JsonValue(std::string value) : value_(std::move(value)) {}
JsonValue::JsonValue(object_t value) : value_(std::move(value)) {}
JsonValue::JsonValue(array_t value) : value_(std::move(value)) {}

bool JsonValue::is_null() const { return std::holds_alternative<std::nullptr_t>(value_); }
bool JsonValue::is_bool() const { return std::holds_alternative<bool>(value_); }
bool JsonValue::is_number() const { return std::holds_alternative<double>(value_); }
bool JsonValue::is_string() const { return std::holds_alternative<std::string>(value_); }
bool JsonValue::is_object() const { return std::holds_alternative<object_t>(value_); }
bool JsonValue::is_array() const { return std::holds_alternative<array_t>(value_); }

bool JsonValue::as_bool() const {
    if (!is_bool()) {
        throw std::runtime_error("JSON value is not a boolean");
    }
    return std::get<bool>(value_);
}

double JsonValue::as_number() const {
    if (!is_number()) {
        throw std::runtime_error("JSON value is not a number");
    }
    return std::get<double>(value_);
}

const std::string& JsonValue::as_string() const {
    if (!is_string()) {
        throw std::runtime_error("JSON value is not a string");
    }
    return std::get<std::string>(value_);
}

const JsonValue::object_t& JsonValue::as_object() const {
    if (!is_object()) {
        throw std::runtime_error("JSON value is not an object");
    }
    return std::get<object_t>(value_);
}

const JsonValue::array_t& JsonValue::as_array() const {
    if (!is_array()) {
        throw std::runtime_error("JSON value is not an array");
    }
    return std::get<array_t>(value_);
}

bool JsonValue::contains(const std::string& key) const {
    if (!is_object()) {
        return false;
    }
    const auto& obj = std::get<object_t>(value_);
    return obj.find(key) != obj.end();
}

const JsonValue& JsonValue::at(const std::string& key) const {
    const auto& obj = as_object();
    auto it = obj.find(key);
    if (it == obj.end()) {
        throw std::out_of_range("JSON object key not found: " + key);
    }
    return it->second;
}

const JsonValue& JsonValue::at(std::size_t index) const {
    const auto& arr = as_array();
    if (index >= arr.size()) {
        throw std::out_of_range("JSON array index out of range");
    }
    return arr[index];
}

std::size_t JsonValue::size() const {
    if (is_object()) {
        return std::get<object_t>(value_).size();
    }
    if (is_array()) {
        return std::get<array_t>(value_).size();
    }
    return 0;
}

const JsonValue::storage_t& JsonValue::storage() const { return value_; }

JsonValue parse_json(const std::string& input) {
    return Parser(input).parse();
}

}  // namespace glfs
