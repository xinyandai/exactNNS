#include <iostream>
#include "include/imisequence.h"
#include "include/util.h"

template <typename DATATYPE>
int execute(const string& baseFormat, const string& dataFile, const string& queryFile) {

    // 1. load retrieved data and query data
    lshbox::Matrix<DATATYPE> data;
    lshbox::Matrix<DATATYPE> query;
    if (baseFormat == "fvecs") {
        lshbox::loadFvecs(data, dataFile);
        lshbox::loadFvecs(query, queryFile);
    } else {
        std::cout << "wrong input data format " << baseFormat << std::endl;
    }

    // 2. build indexes with PQ


    // 3. query
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}