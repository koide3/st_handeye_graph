#ifndef ST_HANDEYE_PARAMETER_XYZ_HPP
#define ST_HANDEYE_PARAMETER_XYZ_HPP

#include <g2o/core/optimizable_graph.h>

namespace g2o {

// 3D point of the calibration pattern
class ParameterPointXYZ : public g2o::Parameter {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ParameterPointXYZ(const Eigen::Vector3d& point)
        : _point(point)
    {}

    ~ParameterPointXYZ() {}

    void setPoint(const Eigen::Vector3d& pt) {
        _point = pt;
    }

    const Eigen::Vector3d& point() const {
        return _point;
    }

    virtual bool read(std::istream &is) override {
        is >> _point[0] >> _point[1] >> _point[2];
        return !is.fail();
    }

    virtual bool write(std::ostream &os) const override {
        os << _point[0] << " " << _point[1] << " " << _point[2];
        return os.good();
    }

protected:
    Eigen::Vector3d _point;
};

}

#endif
