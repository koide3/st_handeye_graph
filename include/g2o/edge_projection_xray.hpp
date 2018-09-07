#ifndef ST_HANDEYE_EDGE_PROJECTION_XRAY_HPP
#define ST_HANDEYE_EDGE_PROJECTION_XRAY_HPP

#include <g2o/core/base_multi_edge.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/slam3d/parameter_camera.h>
#include <g2o/parameter_point_xyz.hpp>


namespace g2o {

// source-detector projection model
// m = point observation
// p[0] = camera params
// p[1] = point params
// v[0] = object2world
// v[1] = object2detector
// v[2] = source_in_world
// error = proj(camera_matrix, point, object2detector * object2world.inverse() * source_in_world)
class EdgeProjectionXray : public g2o::BaseMultiEdge<2, Eigen::Vector2d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeProjectionXray() :
        g2o::BaseMultiEdge<2, Eigen::Vector2d>()
    {
        resize(3);

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
        // v[0] = object2world
        // v[1] = object2detector
        // v[2] = source_in_world
        // error = proj(camera_matrix, point, object2detector * object2world.inverse() * source_in_world)

        const g2o::VertexSE3* v_object2world = dynamic_cast<const g2o::VertexSE3*>(_vertices[0]);
        const g2o::VertexSE3* v_object2detector = dynamic_cast<const g2o::VertexSE3*>(_vertices[1]);
        const g2o::VertexPointXYZ* v_source_in_world = dynamic_cast<const g2o::VertexPointXYZ*>(_vertices[2]);

        Eigen::Isometry3d object2world = v_object2world->estimate();
        Eigen::Isometry3d object2detector = v_object2detector->estimate();
        Eigen::Vector3d pt = object2detector * ptparam->point();
        Eigen::Vector3d source = object2detector * object2world.inverse() * v_source_in_world->estimate();

        double t = source.z() / (source.z() - pt.z());
        Eigen::Vector3d projected = (source + (pt - source) * t);
        // for homogeneous transformation
        projected.z() = 1.0;

        Eigen::Vector3d uv1 = camparam->Kcam() * projected;

        _error = _measurement - uv1.head<2>();
    }

private:
    g2o::ParameterCamera* camparam;
    g2o::ParameterPointXYZ* ptparam;
};

}


#endif
