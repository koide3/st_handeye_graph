#ifndef ST_HANDEYE_EDGE_PROJECTION_PINHOLE_HPP
#define ST_HANDEYE_EDGE_PROJECTION_PINHOLE_HPP

#include <g2o/core/base_unary_edge.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/slam3d/parameter_camera.h>
#include <g2o/parameter_point_xyz.hpp>


namespace g2o {
// pinhole projection model
// m = point observation
// p[0] = camera param
// p[1] = point param
// v[0] = object2eye
// error = proj(camera_matrix, object2eye * point) - measurement
class EdgeProjectionPinhole : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeProjectionPinhole() :
        g2o::BaseUnaryEdge<2, g2o::Vector2D, g2o::VertexSE3>()
    {
        resizeParameters(2);
        installParameter(camparam, 0);
        installParameter(ptparam, 1);
    }

    virtual bool read(std::istream &is) override {
        Eigen::Vector2d v;
        is >> v(0) >> v(1);
        _measurement = v;

        for (int i = 0; i < information().rows(); ++i)
            for (int j = i; j < information().cols(); ++j) {
                is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
        }
        return true;
    }

    virtual bool write(std::ostream &os) const override {
        Eigen::Vector2d v = _measurement;
        os << v(0) << " " << v(1);
        for (int i = 0; i < information().rows(); ++i)
            for (int j = i; j < information().cols(); ++j)
                os << " " << information()(i, j);
        return os.good();
    }

    virtual void computeError() override {
        const g2o::VertexSE3* v_object2eye = dynamic_cast<const g2o::VertexSE3*>(_vertices[0]);
        Eigen::Isometry3d object2eye = v_object2eye->estimate();

        Eigen::Vector3d uvs = camparam->Kcam() * object2eye * ptparam->point();
        Eigen::Vector2d uv = uvs.head<2>() / uvs[2];

        _error = _measurement - uv;
    }

private:
    g2o::ParameterCamera* camparam;
    g2o::ParameterPointXYZ* ptparam;
};

}


#endif
