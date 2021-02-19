#include <string>
#include <filesystem>
#include <vector>
#include <numeric>
#include <fstream>
#include <stdexcept>

#include <SpidrAnalysis.h>

std::vector<float> readData(const std::string fileName);
void writeData(const std::vector<float> data, const std::string fileName);

int main() {
	// Set data info
	std::filesystem::path projectDir = std::filesystem::current_path().parent_path();
	const std::string fileName = "CheckeredBoxes_2Ch_32.bin";
	const std::string emebddingName = "CheckeredBoxes_2Ch_32_sp-tSNE_Chamfer.bin";

	// load data
	const std::vector<float> data = readData(projectDir.string() + "/data/" + fileName);

	// image data
	const size_t numPoints = 32 * 32;
	const size_t numDims = 2;
	const ImgSize imgSize(32, 32);
	std::vector<unsigned int> pointIDsGlobal(numPoints);
	std::iota(pointIDsGlobal.begin(), pointIDsGlobal.end(), 0);

	// Spidr settings
	SpidrAnalysis spidr;
	const feature_type featureType = feature_type::PCLOUD;
	const knn_library knnLibrary = knn_library::KNN_HNSW;
	const distance_metric distanceMetric = distance_metric::METRIC_CHA;
	const size_t spatialNeighborsInEachDirection = 1;

	// t-SNE settings
	const int numIterations = 1000;
	const int perplexity = 20;
	const int exaggeration = 250;
	const int expDecay = 70;

	// Compute spatially informed embedding
	spidr.setupData(data, pointIDsGlobal, numDims, imgSize, emebddingName);
	spidr.initializeAnalysisSettings(featureType, loc_Neigh_Weighting::WEIGHT_UNIF, spatialNeighborsInEachDirection, 0, knnLibrary, distanceMetric, 0, numIterations, perplexity, exaggeration, expDecay);
	spidr.compute();
	std::vector<float> embedding = spidr.output();

	// Save embedding
	writeData(embedding, projectDir.string() + "/data/" + emebddingName);

    return 0;
}

std::vector<float> readData(const std::string fileName)
{
	std::vector<float> fileContents;

	// open file 
	std::ifstream fin(fileName, std::ios::in | std::ios::binary);
	if (!fin.is_open()) {
		throw std::invalid_argument("Unable to load file: " + fileName);
	}
	else {
		// number of data points
		fin.seekg(0, std::ios::end);
		auto fileSize = fin.tellg();
		auto numDataPoints = fileSize / sizeof(float);
		fin.seekg(0, std::ios::beg);

		// read data
		fileContents.resize(numDataPoints);
		fin.read(reinterpret_cast<char*>(fileContents.data()), fileSize);
		fin.close();
	}

	return fileContents;
}

void writeData(const std::vector<float> data, const std::string fileName) {
	std::ofstream fout(fileName, std::ios::out | std::ios::binary);
	fout.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
	fout.close();
}
