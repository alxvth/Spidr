#include <string>
#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdexcept>

template<typename T>
std::vector<T> loadData(const std::string fileName)
{
	std::vector<T> fileContents;

	// open file 
	std::ifstream istrm(fileName, std::ios::in | std::ios::binary);
	if (!istrm.is_open()) {
		throw std::invalid_argument("Unable to load file: " + fileName);
	}
	else {
		// number of data points
		istrm.seekg(0, istrm.end);
		auto N = istrm.tellg();		
		istrm.seekg(0, istrm.beg);

		// read data
		fileContents.resize(N * sizeof(T));
		istrm.read(reinterpret_cast<char*>(fileContents.data()), fileContents.size() * sizeof(T));
		istrm.close();
	}

	return fileContents;
}

int main() {
	const std::string fileName = "data/CheckeredBoxes_2Ch_32.bin";

	std::vector<float> data = loadData<float>(fileName);

	// Settings

	// Compute embedding

	// Save embedding

    return 0;
}

