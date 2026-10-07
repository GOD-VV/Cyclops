#include <ros/ros.h>
#include <rosbag/bag.h>
#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>

#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <utility>
#include <cstdint>
#include <cmath>

struct PointXYZRGBI {
    PCL_ADD_POINT4D;
    float intensity;
    PCL_ADD_RGB;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZRGBI,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, rgb, rgb)
)

typedef PointXYZRGBI PointType;

// 相机分辨率（原始）
static const int IMAGE_WIDTH = 1280;
static const int IMAGE_HEIGHT = 720;

// bag 输出目录
static const std::string ODOM_TXT_NAME = "odom.txt";
static const std::string ODOM_REF_TXT_NAME = "odom_ref.txt";
static const std::string CAMERA_SUBDIR = "camera";
static const std::string PCD_SUBDIR = "pcd";

static const std::string CAMERA_IMAGE_PREFIX = "camera_image_";
static const std::string CAMERA_IMAGE_EXT = ".png";

static const std::string PCD_PREFIX = "accumulated_cloud_";
static const std::string PCD_EXT = ".pcd";

static const double scale_factor = 720.0 / 256.0;
static const int downsampled_width = static_cast<int>(IMAGE_WIDTH / scale_factor);   // 455
static const int downsampled_height = static_cast<int>(IMAGE_HEIGHT / scale_factor); // 256

static bool parseOdomTxt(const std::string& odom_file, std::vector<Eigen::Matrix4d>& poses) {
    std::ifstream in(odom_file);
    if (!in.is_open()) return false;

    poses.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        std::stringstream ss(line);
        double x, y, z, qw, qx, qy, qz;
        if (!(ss >> x >> y >> z >> qw >> qx >> qy >> qz)) continue;

        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T(0, 3) = x;
        T(1, 3) = y;
        T(2, 3) = z;
        Eigen::Quaterniond q(qw, qx, qy, qz);
        T.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
        poses.push_back(T);
    }
    return !poses.empty();
}

static inline Eigen::Matrix4d startPoseForFrame(
    int frame_idx,
    const std::vector<Eigen::Matrix4d>& poses_end,
    const std::vector<Eigen::Matrix4d>& poses_ref) {
    if (!poses_ref.empty() && frame_idx < static_cast<int>(poses_ref.size())) {
        return poses_ref[frame_idx];
    }
    if (frame_idx > 0 && frame_idx - 1 < static_cast<int>(poses_end.size())) {
        return poses_end[frame_idx - 1];
    }
    return poses_end.empty() ? Eigen::Matrix4d::Identity() : poses_end[0];
}

static void initCalibration(cv::Mat& intrinsic_matrix,
                             cv::Mat& distortion_coeffs,
                             cv::Mat& extrinsic_matrix,
                             cv::Mat& adjusted_intrinsic_matrix) {
    intrinsic_matrix = (cv::Mat_<double>(3, 3) <<
        908.524, 0.0, 642.150,
        0.0, 908.919, 352.982,
        0.0, 0.0, 1.0
    );

    distortion_coeffs = (cv::Mat_<double>(1, 5) <<
        -0.055537205189466476, 0.06588292121887207, -0.0013024559011682868, -0.0010083256056532264, -0.02116576768457889
    );

    // 外参（LiDAR -> Camera）
    extrinsic_matrix = (cv::Mat_<double>(4, 4) <<
        0.00208164,  -0.999858,  -0.0167461,  0.0488741,
        0.430466,    0.0160109,  -0.902465,   -0.0482999,
        0.902605,   -0.00532965,  0.430438,   -0.044622,
         0.0,        0.0,         0.0,         1.0
    );

    adjusted_intrinsic_matrix = intrinsic_matrix.clone();
    adjusted_intrinsic_matrix.at<double>(0, 0) /= scale_factor; // fx
    adjusted_intrinsic_matrix.at<double>(1, 1) /= scale_factor; // fy
    adjusted_intrinsic_matrix.at<double>(0, 2) /= scale_factor; // cx
    adjusted_intrinsic_matrix.at<double>(1, 2) /= scale_factor; // cy
}

static inline bool projectToPixel(const Eigen::Vector3d& Pw,
                                    const Eigen::Matrix3d& R,
                                    const Eigen::Vector3d& t,
                                    double fx, double fy, double cx, double cy,
                                    int width, int height,
                                    int& u, int& v) {
    // Pc = R * Pw + t
    Eigen::Vector3d Pc = R * Pw + t;
    const double X = Pc.x();
    const double Y = Pc.y();
    const double Z = Pc.z();
    if (Z <= 0.0) return false;

    const double u_f = (fx * X + cx * Z) / Z;
    const double v_f = (fy * Y + cy * Z) / Z;

    u = static_cast<int>(u_f);
    v = static_cast<int>(v_f);
    if (u < 0 || u >= width || v < 0 || v >= height) return false;
    return true;
}

static inline int parseFrameIndexFromPcdFilename(const std::string& filename) {
    // accumulated_cloud_<frame_index>.pcd
    if (filename.rfind(PCD_PREFIX, 0) != 0) return -1;
    if (filename.size() < PCD_PREFIX.size() + PCD_EXT.size()) return -1;
    if (filename.substr(filename.size() - PCD_EXT.size()) != PCD_EXT) return -1;

    std::string num = filename.substr(
        PCD_PREFIX.size(),
        filename.size() - PCD_PREFIX.size() - PCD_EXT.size()
    );
    try {
        return std::stoi(num);
    } catch (...) {
        return -1;
    }
}

static bool encodeCameraToCompressedImage(const cv::Mat& bgr_image, const ros::Time& stamp,
                                           const std::string& frame_id,
                                           const std::string& format,
                                           sensor_msgs::CompressedImage& out_msg,
                                           int jpeg_quality) {
    if (bgr_image.empty()) return false;

    std::vector<uchar> buf;
    std::vector<int> params;
    if (format == "jpeg") {
        params.push_back(cv::IMWRITE_JPEG_QUALITY);
        params.push_back(jpeg_quality);
    }

    const std::string ext = (format == "jpeg") ? ".jpg" : ".png";
    if (!cv::imencode(ext, bgr_image, buf, params)) return false;

    out_msg.header.stamp = stamp;
    out_msg.header.frame_id = frame_id;
    out_msg.format = format;
    out_msg.data = buf;
    return true;
}

static bool processOneBagDir(const std::string& bag_dir,
                              const std::string& output_bag_path,
                              const std::string& camera_topic,
                              const std::string& point_topic,
                              const std::string& camera_frame_id,
                              const std::string& pointcloud_frame_id,
                              double start_sec,
                              double dt_sec,
                              int jpeg_quality,
                              int frame_stride,
                              bool transform_to_global) {
    const std::string odom_file = (boost::filesystem::path(bag_dir) / ODOM_TXT_NAME).string();
    const std::string odom_ref_file = (boost::filesystem::path(bag_dir) / ODOM_REF_TXT_NAME).string();
    std::vector<Eigen::Matrix4d> poses;
    std::vector<Eigen::Matrix4d> poses_ref;
    if (!parseOdomTxt(odom_file, poses)) {
        ROS_WARN("Skip: failed to read odom: %s", odom_file.c_str());
        return false;
    }
    if (boost::filesystem::exists(odom_ref_file)) {
        parseOdomTxt(odom_ref_file, poses_ref);
    }
    if (transform_to_global && poses_ref.empty()) {
        ROS_WARN_ONCE("No %s: frame 0 uses odom[0] as T_start (first window may be offset ~cm)",
                      ODOM_REF_TXT_NAME.c_str());
    }

    cv::Mat intrinsic_matrix, distortion_coeffs, extrinsic_matrix, adjusted_intrinsic_matrix;
    initCalibration(intrinsic_matrix, distortion_coeffs, extrinsic_matrix, adjusted_intrinsic_matrix);

    Eigen::Matrix3d R_eig;
    Eigen::Vector3d t_eig;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) R_eig(r, c) = extrinsic_matrix.at<double>(r, c);
        t_eig(r) = extrinsic_matrix.at<double>(r, 3);
    }

    const double fx = adjusted_intrinsic_matrix.at<double>(0, 0);
    const double fy = adjusted_intrinsic_matrix.at<double>(1, 1);
    const double cx = adjusted_intrinsic_matrix.at<double>(0, 2);
    const double cy = adjusted_intrinsic_matrix.at<double>(1, 2);

    const boost::filesystem::path pcd_root = boost::filesystem::path(bag_dir) / PCD_SUBDIR;
    const boost::filesystem::path camera_root = boost::filesystem::path(bag_dir) / CAMERA_SUBDIR;

    if (!boost::filesystem::exists(pcd_root) || !boost::filesystem::exists(camera_root)) {
        ROS_WARN("Skip: missing %s/%s dirs in %s", PCD_SUBDIR.c_str(), CAMERA_SUBDIR.c_str(), bag_dir.c_str());
        return false;
    }

    std::vector<std::pair<int, std::string>> frame_pcds;
    for (boost::filesystem::directory_iterator it(pcd_root); it != boost::filesystem::directory_iterator(); ++it) {
        if (!boost::filesystem::is_regular_file(it->path())) continue;
        const std::string filename = it->path().filename().string();
        const int idx = parseFrameIndexFromPcdFilename(filename);
        if (idx < 0) continue;
        frame_pcds.push_back({idx, it->path().string()});
    }
    std::sort(frame_pcds.begin(), frame_pcds.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    if (frame_pcds.empty()) {
            ROS_WARN("Skip: no accumulated_cloud_*.pcd found in %s", pcd_root.string().c_str());
        return false;
    }

    rosbag::Bag bag_out;
    try {
        bag_out.open(output_bag_path, rosbag::bagmode::Write);
    } catch (const std::exception& e) {
        ROS_ERROR("Failed to open output bag: %s (%s)", output_bag_path.c_str(), e.what());
        return false;
    }

    ROS_INFO("Colorize frames in: %s -> %s (transform_to_global=%d)",
             bag_dir.c_str(), output_bag_path.c_str(), transform_to_global);
    if (!transform_to_global) {
        ROS_INFO("  Points stay in PCD local frame (per-frame frame_id suffix)");
    } else {
        ROS_INFO("  Global map: P_world = T_start * P_ref, frame_id='%s' (RViz Fixed Frame = world)",
                 pointcloud_frame_id.c_str());
    }

    int written = 0;
    for (const auto& fp : frame_pcds) {
        const int frame_idx = fp.first;
        const std::string& pcd_path = fp.second;
        if (frame_stride > 1 && (frame_idx % frame_stride) != 0) {
            continue; 
        }
        if (frame_idx < 0 || frame_idx >= static_cast<int>(poses.size())) {
            ROS_WARN("Skip frame %d: odom missing (poses=%zu)", frame_idx, poses.size());
            continue;
        }

        const std::string cam_path = (camera_root / (CAMERA_IMAGE_PREFIX + std::to_string(frame_idx) + CAMERA_IMAGE_EXT)).string();
        if (!boost::filesystem::exists(cam_path)) {
            ROS_WARN("Skip frame %d: camera image not found: %s", frame_idx, cam_path.c_str());
            continue;
        }

        cv::Mat camera_bgr = cv::imread(cam_path, cv::IMREAD_COLOR);
        if (camera_bgr.empty()) {
            ROS_WARN("Skip frame %d: failed to imread: %s", frame_idx, cam_path.c_str());
            continue;
        }

        if (camera_bgr.cols != downsampled_width || camera_bgr.rows != downsampled_height) {
            cv::resize(camera_bgr, camera_bgr, cv::Size(downsampled_width, downsampled_height), 0, 0, cv::INTER_LINEAR);
        }

        pcl::PointCloud<PointType>::Ptr cloud_in(new pcl::PointCloud<PointType>());
        if (pcl::io::loadPCDFile<PointType>(pcd_path, *cloud_in) < 0 || !cloud_in) {
            ROS_WARN("Skip frame %d: failed to load PCD: %s", frame_idx, pcd_path.c_str());
            continue;
        }

        const Eigen::Matrix4d T_end = poses[frame_idx];
        const Eigen::Matrix4d T_start = startPoseForFrame(frame_idx, poses, poses_ref);
        const Eigen::Matrix4d ref_T_end = T_start.inverse() * T_end;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGB>());
        cloud_out->reserve(cloud_in->points.size());

        for (const auto& pt : cloud_in->points) {
            Eigen::Vector4d P_ref(pt.x, pt.y, pt.z, 1.0);

            const Eigen::Vector4d P_end_h = ref_T_end * P_ref;
            Eigen::Vector3d P_lidar_end = P_end_h.head<3>();
            int u = -1, v = -1;
            bool ok = projectToPixel(P_lidar_end, R_eig, t_eig, fx, fy, cx, cy,
                                     downsampled_width, downsampled_height, u, v);

            uint8_t r = 0, g = 0, b = 0;
            if (ok) {
                const cv::Vec3b bgr = camera_bgr.at<cv::Vec3b>(v, u);
                b = static_cast<uint8_t>(bgr[0]);
                g = static_cast<uint8_t>(bgr[1]);
                r = static_cast<uint8_t>(bgr[2]);
            }

            // 全局地图：真实世界坐标 = T_start * P_ref
            Eigen::Vector4d P_out_h = transform_to_global ? (T_start * P_ref) : P_ref;
            pcl::PointXYZRGB out_pt;
            out_pt.x = static_cast<float>(P_out_h.x());
            out_pt.y = static_cast<float>(P_out_h.y());
            out_pt.z = static_cast<float>(P_out_h.z());
            out_pt.r = r;
            out_pt.g = g;
            out_pt.b = b;
            cloud_out->points.push_back(out_pt);
        }

        cloud_out->width = static_cast<uint32_t>(cloud_out->points.size());
        cloud_out->height = 1;
        cloud_out->is_dense = false;

        double t = start_sec + frame_idx * dt_sec;
        if (!std::isfinite(t) || t <= 0.0) t = 1e-3;
        const ros::Time stamp(t);
        sensor_msgs::CompressedImage cam_msg;
        if (!encodeCameraToCompressedImage(camera_bgr, stamp, camera_frame_id, "jpeg", cam_msg, jpeg_quality)) {
            ROS_WARN("Skip frame %d: failed to encode camera image", frame_idx);
            continue;
        }
        bag_out.write(camera_topic, stamp, cam_msg);

        // write pointcloud
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud_out, cloud_msg);
        cloud_msg.header.stamp = stamp;
        cloud_msg.header.frame_id = transform_to_global
            ? pointcloud_frame_id
            : (pointcloud_frame_id + "_" + std::to_string(frame_idx));
        bag_out.write(point_topic, stamp, cloud_msg);

        written++;
    }

    bag_out.close();
    ROS_INFO("Finished %s (written frames: %d)", bag_dir.c_str(), written);
    return written > 0;
}

static void findAllOdomTxtDirs(const std::string& input_base_dir, std::vector<std::string>& bag_dirs) {
    bag_dirs.clear();
    if (!boost::filesystem::exists(input_base_dir) || !boost::filesystem::is_directory(input_base_dir)) return;

    try {
        boost::filesystem::recursive_directory_iterator it(input_base_dir), end;
        for (; it != end; ++it) {
            if (!boost::filesystem::is_regular_file(it->path())) continue;
            if (it->path().filename().string() == ODOM_TXT_NAME) {
                bag_dirs.push_back(it->path().parent_path().string());
            }
        }
    } catch (const std::exception& e) {
        ROS_ERROR("Error while searching odom.txt dirs: %s", e.what());
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        ROS_ERROR("Usage: rosrun lbm_ros colorize_pcd_to_global_bag <processed_output_base_dir> <output_base_dir>");
        ROS_ERROR("This tool finds all */odom.txt, then for each dir expects:");
        ROS_ERROR("  - %s/%s<frame_idx>%s", PCD_SUBDIR.c_str(), PCD_PREFIX.c_str(), PCD_EXT.c_str());
        ROS_ERROR("  - %s/%s<frame_idx>%s", CAMERA_SUBDIR.c_str(), CAMERA_IMAGE_PREFIX.c_str(), CAMERA_IMAGE_EXT.c_str());
        return 1;
    }

    std::string input_base_dir = argv[1];
    std::string output_base_dir = argv[2];

    std::string camera_topic = "/camera/color/image_raw/compressed";
    std::string point_topic = "/colored_pointcloud";
    std::string camera_frame_id = "camera";
    std::string pointcloud_frame_id = "world";

    double start_sec = 1e-3;
    double dt_sec = 0.5;
    int jpeg_quality = 95;
    int frame_stride = 1;
    bool transform_to_global = true;

    ros::init(argc, argv, "colorize_pcd_to_global_bag");
    ros::NodeHandle nh;
    nh.param("camera_topic", camera_topic, camera_topic);
    nh.param("point_topic", point_topic, point_topic);
    nh.param("camera_frame_id", camera_frame_id, camera_frame_id);
    nh.param("pointcloud_frame_id", pointcloud_frame_id, pointcloud_frame_id);
    nh.param("start_sec", start_sec, start_sec);
    nh.param("dt_sec", dt_sec, dt_sec);
    nh.param("jpeg_quality", jpeg_quality, jpeg_quality);
    nh.param("frame_stride", frame_stride, frame_stride);
    nh.param("transform_to_global", transform_to_global, transform_to_global);

    // 查找所有 bag_dirs
    std::vector<std::string> bag_dirs;
    findAllOdomTxtDirs(input_base_dir, bag_dirs);
    if (bag_dirs.empty()) {
        ROS_ERROR("No odom.txt found under: %s", input_base_dir.c_str());
        return 1;
    }

    boost::filesystem::path input_base_path(input_base_dir);
    boost::filesystem::path output_base_path(output_base_dir);

    int total = static_cast<int>(bag_dirs.size());
    int done = 0;
    for (int i = 0; i < total; ++i) {
        const std::string bag_dir = bag_dirs[i];
        boost::filesystem::path bag_dir_path(bag_dir);
        boost::filesystem::path rel_dir;
        try {
            rel_dir = boost::filesystem::relative(bag_dir_path, input_base_path);
        } catch (...) {
            rel_dir = bag_dir_path.filename();
        }

        boost::filesystem::path out_dir = output_base_path / rel_dir;
        boost::filesystem::create_directories(out_dir.string());
        const std::string out_bag_path = (out_dir / "camera_and_colored_pointcloud.bag").string();

        ROS_INFO("(%d/%d) Processing: %s", i + 1, total, bag_dir.c_str());
        if (processOneBagDir(bag_dir, out_bag_path,
                             camera_topic, point_topic,
                             camera_frame_id, pointcloud_frame_id,
                             start_sec, dt_sec, jpeg_quality,
                             frame_stride, transform_to_global)) {
            done++;
        }
    }

    ROS_INFO("All done. Success: %d/%d", done, total);
    return 0;
}

