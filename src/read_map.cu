#include "read_map.h"

std::vector<Wall> read_map(const std::string& filename){
    
    std::vector<Wall> map;
    
    std::ifstream file(filename);
    if(!file){
        std::cout << "Error reading map file!" << std::endl;
    }

    std::string line;
    point3 wall_components[4];
    float exc;
    while(!file.eof()){
        file.ignore(std::numeric_limits<std::streamsize>::max(), '{');
        file.get(); //We jump the next char, a newline.

        // We now can read each point.
        size_t i=0;
        while (std::getline(file, line)){
            if(line.compare("}") == 0) break;

            // Gets rid of leading spaces or tabs.
            line.erase(0, line.find_first_not_of(" \t"));

            //TODO: cleanup?
            if (i < 4) {
                wall_components[i] = cleanup_point_info(line);
            }
            else {
                exc = std::stof(line.substr(2));
            }
            i++;
        }
        
        map.emplace_back(wall_components, exc);
    }

    return map;
}

point3 cleanup_point_info(const std::string& raw_point_info){
    /*
        Converts a string of components to separate components.
    */

    float components[3];

    // We split the strings by commas.
    std::string split;
    std::istringstream ss(raw_point_info);

    size_t i = 0;
    while(std::getline(ss, split, ',')){
        components[i] = std::stof(split);
        i++;
    }

    return {components};
}