#ifndef ARMA_PYBIND11_H
#define ARMA_PYBIND11_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>

namespace py = pybind11;
namespace pybind11 {
namespace detail {

// Type caster for arma:mat
template <> struct type_caster<arma::mat> {
public:
    PYBIND11_TYPE_CASTER(arma::mat, _("numpy.ndarray"));

    bool load(handle src, bool) {
        if (!isinstance<array_t<double>>(src)) {
            return false;
        }
        py::array_t<double> buf = py::array_t<double>::ensure(src);
        if (!buf) {
            return false;
        }
        if (buf.ndim() != 2) {
            return false;
        }

        size_t rows = buf.shape(0);
        size_t cols = buf.shape(1);

        // Check if the array is Fortran-contiguous
        if (buf.strides(0) != sizeof(double) || buf.strides(1) != sizeof(double)*rows) {
            return false;
        }

        value = arma::mat(rows, cols);
        std::memcpy(value.memptr(), buf.data(), rows * cols * sizeof(double));
        return true;
    }

    // Conversion from C++ to Python
    static handle cast(const arma::mat &src, return_value_policy, handle) {
        // Define shape and strides with explicit casts to py::ssize_t
        std::vector<py::ssize_t> shape = {
            static_cast<py::ssize_t>(src.n_rows),
            static_cast<py::ssize_t>(src.n_cols)
        };
        std::vector<py::ssize_t> strides = {
            static_cast<py::ssize_t>(sizeof(double)),
            static_cast<py::ssize_t>(sizeof(double) * src.n_rows)
        };

        // Create a NumPy array without copying the data
        py::array_t<double> array(shape, strides, src.memptr(), py::none());

        // Make a copy to ensure ownership in Python
        py::array_t<double> copy = array.attr("copy")();

        // Return the copied array as a handle
        return copy.release();
    }
};

// Type caster for arma::vec
template <> struct type_caster<arma::vec> {
public:
    PYBIND11_TYPE_CASTER(arma::vec, _("numpy.ndarray"));

    // Conversion from Python to C++
    bool load(handle src, bool) {
		//std::cout << "Type caster called for arma::vec with type: " << pybind11::str(src.get_type()) << std::endl;
        if (!isinstance<array_t<double>>(src)) {
			std::cout << "Not a numpy array of doubles." << std::endl;
            return false;
        }
        py::array_t<double> buf = py::array_t<double>::ensure(src);
        if (!buf) {
			std::cout << "Failed to ensure numpy array." << std::endl;
            return false;
        }
        
        if (buf.ndim() != 1) {
			std::cout << "Array is not 1-dimensional." << std::endl;
            return false;
        }

        // Retrieve the size of the NumPy array and initialize arma::vec to match
        size_t size = buf.shape(0);
        value = arma::vec(size);
        
        std::memcpy(value.memptr(), buf.data(), size * sizeof(double));
		//std::cout << "Successfully casted to arma::vec with size: " << size << std::endl;
        return true;
    }

    // c++ to python
    static handle cast(const arma::vec &src, return_value_policy, handle) {
        // Define shape and strides with explicit casts to py::ssize_t
        std::vector<py::ssize_t> shape = {
            static_cast<py::ssize_t>(src.n_elem)
        };
        std::vector<py::ssize_t> strides = {
            static_cast<py::ssize_t>(sizeof(double))
        };

        // Create a NumPy array without copying the data
        py::array_t<double> array(shape, strides, src.memptr(), py::none());

        // Make a copy to ensure ownership in Python
        py::array_t<double> copy = array.attr("copy")();

        // Return the copied array as a handle
        return copy.release();
    }
};

template <typename T>
struct type_caster<std::vector<std::vector<T>>> {
public:
    PYBIND11_TYPE_CASTER(std::vector<std::vector<T>>, _("List[List[" PYBIND11_STRINGIFY(T) "]]"));

    // Conversion from Python to C++ is not implemented
    bool load(handle src, bool) {
        return false;
    }

    // Conversion from C++ to Python
    static handle cast(const std::vector<std::vector<T>>& src, return_value_policy /* policy */, handle /* parent */) {

        if (src.empty()) {
            // Return an empty 2D array
            // Define shape as (0, 0)
            std::vector<py::ssize_t> shape = {0, 0};
            // Define strides as {sizeof(T), sizeof(T)*0} which simplifies to {sizeof(T), 0}
            std::vector<py::ssize_t> strides = {static_cast<py::ssize_t>(sizeof(T)), static_cast<py::ssize_t>(sizeof(T) * 0)};
            py::array_t<T> empty_array(shape, strides, nullptr, py::none());
            return empty_array.release();
        }

        size_t rows = src.size();
        size_t cols = src[0].size();

        // Check that all inner vectors have the same size
        for (size_t i = 1; i < rows; ++i) {
            if (src[i].size() != cols) {
                throw std::runtime_error("All inner vectors must have the same size");
            }
        }

        // Define shape and strides with explicit casts to py::ssize_t
        std::vector<py::ssize_t> shape = {
            static_cast<py::ssize_t>(rows),
            static_cast<py::ssize_t>(cols)
        };
        std::vector<py::ssize_t> strides = {
            static_cast<py::ssize_t>(sizeof(T)),
            static_cast<py::ssize_t>(sizeof(T) * rows)
        };

        // Allocate memory for the NumPy array
        py::array_t<T> array(shape, strides, nullptr, py::none());

        // Request buffer info to get the data pointer
        auto buffer = array.request();
        T* ptr = static_cast<T*>(buffer.ptr);

        // Copy data into the NumPy array
        for (size_t i = 0; i < rows; ++i) {
            std::copy(src[i].begin(), src[i].end(), ptr + i * cols);
        }

        // Make a deep copy to ensure Python owns its data
        py::array_t<T> copy = array.attr("copy")();

        // Release the copied array as a handle
        return copy.release();
    }
};


}  // namespace detail
}  // namespace pybind11

#endif  // ARMA_PYBIND11_H
