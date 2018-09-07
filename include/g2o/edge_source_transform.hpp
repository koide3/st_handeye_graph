#ifndef ST_HANDEYE_EDGE_SOURCE_TRANSFORM_HPP
#define ST_HANDEYE_EDGE_SOURCE_TRANSFORM_HPP

#include <g2o/core/base_binary_edge.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>

namespace g2o {

// transform source point to the world frame
// m = world2hand_source
// v[0] = hand2source
// v[1] = source_in_world = world2hand_source.inverse() * -hand2source
class EdgeSourceTransform : public g2o::BaseBinaryEdge<3, Eigen::Isometry3d, g2o::VertexPointXYZ, g2o::VertexPointXYZ> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSourceTransform() :
        g2o::BaseBinaryEdge<3, Eigen::Isometry3d, g2o::VertexPointXYZ, g2o::VertexPointXYZ>()
    {}

    virtual bool read(std::istream &is) override {
        Vector6d v;
        is >> v(0) >> v(1) >> v(2) >> v(3) >> v(4) >> v(5);
        _measurement = g2o::internal::fromVectorMQT(v);

        for (int i = 0; i < information().rows(); ++i)
            for (int j = i; j < information().cols(); ++j) {
                is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
        }
        return true;
    }

    virtual bool write(std::ostream &os) const override {
        Vector6d v = g2o::internal::toVectorMQT(_measurement);
        os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " " << v(4) << " " << v(5) << " ";
        for (int i = 0; i < information().rows(); ++i)
            for (int j = i; j < information().cols(); ++j)
                os << " " << information()(i, j);
        return os.good();
    }

    virtual void computeError() override {
        const g2o::VertexPointXYZ* v_hand2source = dynamic_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
        const g2o::VertexPointXYZ* v_source_in_world = dynamic_cast<const g2o::VertexPointXYZ*>(_vertices[1]);

        Eigen::Vector3d hand2source = v_hand2source->estimate();
        Eigen::Isometry3d world2hand_source = _measurement;
        Eigen::Vector3d source_in_world = world2hand_source.inverse() * -hand2source;

        _error = v_source_in_world->estimate() - source_in_world;
    }
};

}

#endif
