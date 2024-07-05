#include "confini.hpp"
#include <algorithm>  // For std::remove_if
#include <cctype>     // For std::isspace


ConfigIni::ConfigIni(const std::string& filename) : filename(filename) {}

bool ConfigIni::load() {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return false;
    }

    std::string line, section;
    while (getline(file, line)) {
        line.erase(remove_if(line.begin(), line.end(), isspace), line.end()); // Remove whitespace
        if (line.empty() || line[0] == ';' || line[0] == '#') continue; // Skip comments and empty lines

        if (line[0] == '[') {
            section = line.substr(1, line.find(']') - 1);
        } else if (!section.empty()) {
            size_t equals = line.find('=');
            if (equals != std::string::npos) {
                std::string key = line.substr(0, equals);
                std::string value = line.substr(equals + 1);
                data[section][key] = value;
            }
        }
    }
    file.close();
    return true;
}

std::string ConfigIni::getValue(const std::string& section, const std::string& key) const {
    auto sectionIt = data.find(section);
    if (sectionIt != data.end()) {
        auto keyIt = sectionIt->second.find(key);
        if (keyIt != sectionIt->second.end()) {
            return keyIt->second;
        }
    }
    return ""; // Return an empty string if the key or section is not found
}

