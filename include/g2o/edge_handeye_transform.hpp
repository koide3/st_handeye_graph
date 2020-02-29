#ifndef ST_HANDEYE_EDGE_HANDEYE_TRANSFORM_HPP
#define ST_HANDEYE_EDGE_HANDEYE_TRANSFORM_HPP

#include <g2o/core/base_multi_edge.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>


namespace g2o {
// estimate object2eye transformation
// m = world2hand
// v[0] = hand2eye
// v[1] = object2world
// v[2] = object2eye = hand2eye * world2hand * object2world
typedef Eigen::Matrix<double,6,1> Vector6d;
    
class EdgeHandeyeTransform : public g2o::BaseMultiEdge<6, Eigen::Isometry3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeHandeyeTransform() : g2o::BaseMultiEdge<6, Eigen::Isometry3d>()
    {
        resize(3);
    }

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
        const g2o::VertexSE3* v_hand2eye = dynamic_cast<const g2o::VertexSE3*>(_vertices[0]);
        const g2o::VertexSE3* v_object2world = dynamic_cast<const g2o::VertexSE3*>(_vertices[1]);
        const g2o::VertexSE3* v_object2eye = dynamic_cast<const g2o::VertexSE3*>(_vertices[2]);

        Eigen::Isometry3d hand2eye = v_hand2eye->estimate();
        Eigen::Isometry3d object2world = v_object2world->estimate();
        Eigen::Isometry3d object2eye = v_object2eye->estimate();

        Eigen::Isometry3d hand2world = object2world * object2eye.inverse() * hand2eye;
        Eigen::Isometry3d world2hand = hand2world.inverse();
        Eigen::Isometry3d delta = _measurement.inverse() * world2hand;
        _error = g2o::internal::toVectorMQT(delta);
    }

    Eigen::Isometry3d estimated_world2hand() const {
        const g2o::VertexSE3* v_hand2eye = dynamic_cast<const g2o::VertexSE3*>(_vertices[0]);
        const g2o::VertexSE3* v_object2world = dynamic_cast<const g2o::VertexSE3*>(_vertices[1]);
        const g2o::VertexSE3* v_object2eye = dynamic_cast<const g2o::VertexSE3*>(_vertices[2]);

        Eigen::Isometry3d eye2hand = v_hand2eye->estimate().inverse();
        Eigen::Isometry3d world2object = v_object2world->estimate().inverse();
        Eigen::Isometry3d object2eye = v_object2eye->estimate();

        return eye2hand * object2eye * world2object;
    }
};
}


#endif
