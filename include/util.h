#pragma once
#include <iostream>
#include <string>
#include <cstring>
#include <unordered_map>
#include <fstream>
#include <utility>
#include <random>
#include <thread>
#include <functional>
#include <cassert>

#include "matrix.h"

#pragma once
using std::vector;
using std::string;
using std::unordered_map;


namespace lshbox {

    unordered_map<string, string> parseParams(int argc, const char** argv) {
        unordered_map<string, string> params;
        for (int i = 1; i < argc; ++i) {
            const char* pair = argv[i];
            int length = strlen(pair);
            int sepIdx = -1;
            for (int idx = 0; idx < strlen(pair); ++idx) {
                if (pair[idx] == '=')
                    sepIdx = idx;
            }
            if (strlen(pair) < 3 || pair[0] != '-' || pair[1] != '-' || sepIdx == -1) {
                std::cout << "arguments error, format should be --[key]=[value]" << std::endl;
                assert(false);
            }
            string key(pair, 2, sepIdx - 2);
            string value(pair + sepIdx + 1);
            params[key] = value;
        }
        return params;
    }

    template<typename DATATYPE>
    void loadFvecs(lshbox::Matrix<DATATYPE>& data, const string& dataFile) {
        std::ifstream fin(dataFile.c_str(), std::ios::binary | std::ios::ate);
        if (!fin) {
            std::cout << "cannot open file " << dataFile.c_str() << std::endl;
            assert(false);
        }
        unsigned fileSize = fin.tellg();
        fin.seekg(0, fin.beg);
        assert(fileSize != 0);

        int dimension;
        fin.read((char*)&dimension, sizeof(int));
        unsigned bytesPerRecord = dimension * sizeof(DATATYPE) + 4;
        assert(fileSize % bytesPerRecord == 0);
        int cardinality = fileSize / bytesPerRecord;

        data.reset(dimension, cardinality);
        fin.read((char *)(data.getData()), sizeof(float) * dimension);

        int dim;
        for (int i = 1; i < cardinality; ++i) {
            fin.read((char*)&dim, sizeof(int));
            assert(dim == dimension);
            fin.read((char *)(data.getData() + i * dimension), sizeof(float) * dimension);
        }
        fin.close();
    }


};

