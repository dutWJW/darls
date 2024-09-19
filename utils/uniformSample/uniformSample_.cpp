#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <unordered_map>

namespace py = pybind11;
using namespace std;

vector<int> uniformSample(py::array_t<float>& input_array, py::array_t<float>& min_, py::array_t<float>& max_, float sampleDl)
{
    py::buffer_info req = input_array.request();
    float *ptr = static_cast<float *>(req.ptr);
    py::buffer_info req1 = min_.request();
    float *ptr1 = static_cast<float *>(req1.ptr);
    py::buffer_info req2 = max_.request();
    float *ptr2 = static_cast<float *>(req2.ptr);

    size_t mapIdx;
    size_t h = req.shape[0];
	unordered_map<size_t, int> data;
    vector<int> retIndex;
    size_t xIndex, yIndex, zIndex;
    float xMin = *(ptr1 + 0);
    float xMax = *(ptr2 + 0);
    float yMin = *(ptr1 + 1);
    float yMax = *(ptr2 + 1);
    float zMin = *(ptr1 + 2);
    float zMax = *(ptr2 + 2);
	size_t sampleNX = (size_t)floor((xMax - xMin) / sampleDl) + 1;
	size_t sampleNY = (size_t)floor((yMax - yMin) / sampleDl) + 1;
    float tmpX, tmpY, tmpZ;

    for(int i = 0;i < h;++i){
        tmpX = *(ptr + i * 3    );
        tmpY = *(ptr + i * 3 + 1);
        tmpZ = *(ptr + i * 3 + 2);

        xIndex = (size_t)floor((tmpX - xMin) / sampleDl);
        yIndex = (size_t)floor((tmpY - yMin) / sampleDl);
        zIndex = (size_t)floor((tmpZ - zMin) / sampleDl);
        mapIdx = xIndex + sampleNX * yIndex + sampleNX * sampleNY * zIndex;

		if (data.count(mapIdx) < 1)
        {
            data.emplace(mapIdx, 0);
            retIndex.push_back(i);
        }
    }
    return retIndex;
}

PYBIND11_MODULE(uniformSample, m) {

    m.doc() = "uniform sample point cloud";
    m.def("uniformSample", &uniformSample);

}