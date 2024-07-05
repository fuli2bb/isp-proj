#ifndef CONFINI_HPP
#define CONFINI_HPP

#include <string>
#include <map>
#include <fstream>
#include <iostream>

class ConfigIni {
public:
    ConfigIni(const std::string& filename);
    bool load();
    std::string getValue(const std::string& section, const std::string& key) const;

private:
    std::string filename;
    std::map<std::string, std::map<std::string, std::string>> data;
};

#endif // CONFINI_HPP

