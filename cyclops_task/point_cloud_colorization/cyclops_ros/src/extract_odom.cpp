#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <livox_ros_driver2/CustomMsg.h>
#include <sensor_msgs/CompressedImage.h>
#include <nav_msgs/Odometry.h>
#include <boost/filesystem.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>

const int ACCUMULATE_FRAMES = 10;

static bool topicLooksLikeLidar(const std::string& topic) {
    return topic.find("livox") != std::string::npos && topic.find("lidar") != std::string::npos;
}

static bool topicLooksLikeOdom(const std::string& topic) {
    return topic.find("Odometry") != std::string::npos ||
           topic == "/odom" ||
           topic.find("/odom") != std::string::npos;
}

// 从 bag 里自动找 lidar / odom 话题（兼容 process_rosbags 录制结果）
static void detectTopicsFromBag(rosbag::Bag& bag,
                                std::string& lidar_topic,
                                std::string& odom_topic) {
    lidar_topic.clear();
    odom_topic.clear();
    rosbag::View summary_view(bag);
    const std::vector<const rosbag::ConnectionInfo*> connections = summary_view.getConnections();
    for (const rosbag::ConnectionInfo* c : connections) {
        if (!c) continue;
        const std::string& topic = c->topic;
        if (lidar_topic.empty() && topicLooksLikeLidar(topic)) {
            lidar_topic = topic;
        }
        if (odom_topic.empty() && topicLooksLikeOdom(topic)) {
            odom_topic = topic;
        }
    }
}

static inline Eigen::Matrix4d odomMsgToMatrix(const nav_msgs::Odometry::ConstPtr& odometry_msg) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(0, 3) = odometry_msg->pose.pose.position.x;
    T(1, 3) = odometry_msg->pose.pose.position.y;
    T(2, 3) = odometry_msg->pose.pose.position.z;
    Eigen::Quaterniond q(
        odometry_msg->pose.pose.orientation.w,
        odometry_msg->pose.pose.orientation.x,
        odometry_msg->pose.pose.orientation.y,
        odometry_msg->pose.pose.orientation.z
    );
    T.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    return T;
}

static void appendMatrixToFile(std::ofstream& out_file, const Eigen::Matrix4d& T) {
    if (!out_file.is_open()) return;
    Eigen::Quaterniond q(T.block<3, 3>(0, 0));
    q.normalize();
    out_file << std::fixed << std::setprecision(15);
    out_file << T(0, 3) << " " << T(1, 3) << " " << T(2, 3) << " "
             << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << "\n";
}

void appendOdomToFile(std::ofstream& out_file, const nav_msgs::Odometry::ConstPtr& odom_msg) {
    if (!out_file.is_open()) {
        ROS_ERROR("File stream is not open");
        return;
    }
    
    out_file << std::fixed << std::setprecision(15);
    out_file << odom_msg->pose.pose.position.x << " "
             << odom_msg->pose.pose.position.y << " "
             << odom_msg->pose.pose.position.z << " "
             << odom_msg->pose.pose.orientation.w << " "
             << odom_msg->pose.pose.orientation.x << " "
             << odom_msg->pose.pose.orientation.y << " "
             << odom_msg->pose.pose.orientation.z << "\n";
}

void processBagFile(const std::string& bag_file, const std::string& input_base_dir, const std::string& output_base_dir) {
    ROS_INFO("Processing bag file: %s", bag_file.c_str());
    
    boost::filesystem::path bag_path(bag_file);
    boost::filesystem::path input_base_path(input_base_dir);
    
    boost::filesystem::path relative_path;
    try {
        relative_path = boost::filesystem::relative(bag_path, input_base_path);
        
        if (relative_path.empty()) {
            std::string bag_file_str = boost::filesystem::path(bag_file).string();
            std::string input_base_str = boost::filesystem::path(input_base_dir).string();
            
            std::replace(bag_file_str.begin(), bag_file_str.end(), '\\', '/');
            std::replace(input_base_str.begin(), input_base_str.end(), '\\', '/');
            
            if (bag_file_str.find(input_base_str) == 0) {
                std::string relative_str = bag_file_str.substr(input_base_str.length());
                while (!relative_str.empty() && (relative_str[0] == '/' || relative_str[0] == '\\')) {
                    relative_str = relative_str.substr(1);
                }
                if (!relative_str.empty()) {
                    relative_path = relative_str;
                } else {
                    relative_path = bag_path.filename();
                }
            } else {
                relative_path = bag_path.filename();
            }
        }
    } catch (const std::exception& e) {
        ROS_WARN("Failed to compute relative path, using filename only: %s", e.what());
        relative_path = bag_path.filename();
    }
    
    std::string bag_name = relative_path.stem().string();
    
    boost::filesystem::path relative_parent = relative_path.parent_path();
    
    boost::filesystem::path output_path = boost::filesystem::path(output_base_dir) / relative_parent / bag_name;
    std::string output_dir = output_path.string();
    
    boost::filesystem::create_directories(output_dir);
    ROS_INFO("Output directory: %s", output_dir.c_str());
    
    std::string odom_file = output_dir + "/odom.txt";
    std::string odom_ref_file = output_dir + "/odom_ref.txt";
    std::ofstream odom_out_file(odom_file);
    std::ofstream odom_ref_out_file(odom_ref_file);
    if (!odom_out_file.is_open()) {
        ROS_ERROR("Failed to open odom output file: %s", odom_file.c_str());
        return;
    }
    if (!odom_ref_out_file.is_open()) {
        ROS_ERROR("Failed to open odom ref output file: %s", odom_ref_file.c_str());
        odom_out_file.close();
        return;
    }
    
    odom_out_file << "# Odom end pose per saved frame: [x y z qw qx qy qz]\n";
    odom_ref_out_file << "# Odom ref pose at start of each accumulate window (PCD frame)\n";
    
    rosbag::Bag bag;
    try {
        bag.open(bag_file, rosbag::bagmode::Read);
    } catch (const std::exception& e) {
        ROS_ERROR("Failed to open bag file %s: %s", bag_file.c_str(), e.what());
        odom_out_file.close();
        odom_ref_out_file.close();
        return;
    }

    std::string lidar_topic;
    std::string odom_topic;
    detectTopicsFromBag(bag, lidar_topic, odom_topic);

    if (lidar_topic.empty()) lidar_topic = "/livox/lidar";
    if (odom_topic.empty()) odom_topic = "/Odometry";

    bool count_on_odom = lidar_topic.empty();
    ROS_INFO("  lidar topic: %s", lidar_topic.c_str());
    ROS_INFO("  odom topic:  %s", odom_topic.c_str());

    std::vector<std::string> topics = {lidar_topic, odom_topic};
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    bool ref_ready = false;
    Eigen::Matrix4d world_T_accum_ref = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d world_T_this = Eigen::Matrix4d::Identity();
    nav_msgs::Odometry::ConstPtr current_odom_msg = nullptr;

    int frame_count = 0;
    int accumulated_frames = 0;
    int lidar_msg_count = 0;
    int odom_msg_count = 0;
    int odom_instantiate_fail = 0;

    auto trySaveAccumWindow = [&]() {
        if (current_odom_msg == nullptr) {
            ROS_WARN("No odom message available for frame %d", frame_count);
            return;
        }
        appendMatrixToFile(odom_ref_out_file, world_T_accum_ref);
        appendOdomToFile(odom_out_file, current_odom_msg);
        frame_count++;
        world_T_accum_ref = world_T_this;
        accumulated_frames = 0;
    };

    for (const rosbag::MessageInstance& m : view) {
        const std::string& topic = m.getTopic();

        if (topic == odom_topic) {
            nav_msgs::Odometry::ConstPtr odometry_msg = m.instantiate<nav_msgs::Odometry>();
            if (!odometry_msg) {
                odom_instantiate_fail++;
                continue;
            }
            odom_msg_count++;
            world_T_this = odomMsgToMatrix(odometry_msg);
            current_odom_msg = odometry_msg;

            if (!ref_ready) {
                world_T_accum_ref = world_T_this;
                ref_ready = true;
            }

            if (count_on_odom) {
                accumulated_frames++;
                if (accumulated_frames >= ACCUMULATE_FRAMES) {
                    trySaveAccumWindow();
                }
            }
            continue;
        }

        if (!count_on_odom && topic == lidar_topic && ref_ready) {
            lidar_msg_count++;
            accumulated_frames++;
            if (accumulated_frames >= ACCUMULATE_FRAMES) {
                trySaveAccumWindow();
            }
        }
    }

    if (accumulated_frames > 0 && current_odom_msg != nullptr) {
        appendMatrixToFile(odom_ref_out_file, world_T_accum_ref);
        appendOdomToFile(odom_out_file, current_odom_msg);
        frame_count++;
    }

    if (frame_count == 0 && odom_msg_count == 0) {
        ROS_ERROR("  No valid Odometry parsed. Check topic name / message type in bag.");
    }

    odom_ref_out_file.close();
    odom_out_file.close();
    bag.close();
    ROS_INFO("  stats: lidar_msgs=%d odom_msgs=%d odom_inst_fail=%d",
             lidar_msg_count, odom_msg_count, odom_instantiate_fail);
    ROS_INFO("Finished processing bag file: %s (saved %d odom entries to %s)",
             bag_file.c_str(), frame_count, odom_file.c_str());
}

void findBagFiles(const std::string& directory, std::vector<std::string>& bag_files) {
    if (!boost::filesystem::exists(directory) || !boost::filesystem::is_directory(directory)) {
        ROS_WARN("Directory does not exist or is not a directory: %s", directory.c_str());
        return;
    }

    try {
        for (boost::filesystem::recursive_directory_iterator it(directory); it != boost::filesystem::recursive_directory_iterator(); ++it) {
            if (boost::filesystem::is_regular_file(it->path()) && it->path().extension() == ".bag") {
                bag_files.push_back(it->path().string());
            }
        }
    } catch (const std::exception& e) {
        ROS_ERROR("Error while searching for bag files: %s", e.what());
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        ROS_ERROR("Usage: rosrun lbm_ros extract_odom_node <rosbag_directory> <output_directory>");
        ROS_ERROR("Example: rosrun lbm_ros extract_odom_node /path/to/rosbags /path/to/output");
        return 1;
    }

    std::string bag_directory = argv[1];
    std::string output_base_dir = argv[2];

    ros::init(argc, argv, "extract_odom");
    ros::NodeHandle nh;
    boost::filesystem::create_directories(output_base_dir);

    ROS_INFO("Searching for rosbag files in: %s", bag_directory.c_str());
    std::vector<std::string> bag_files;
    findBagFiles(bag_directory, bag_files);

    if (bag_files.empty()) {
        ROS_ERROR("No rosbag files found in directory: %s", bag_directory.c_str());
        return 1;
    }

    ROS_INFO("Found %zu rosbag file(s)", bag_files.size());

    int total_bags = bag_files.size();
    int processed_bags = 0;
    for (size_t i = 0; i < bag_files.size(); ++i) {
        ROS_INFO("==========================================");
        ROS_INFO("Processing bag [%zu/%zu]: %s", i + 1, bag_files.size(), bag_files[i].c_str());
        ROS_INFO("==========================================");
        
        try {
            processBagFile(bag_files[i], bag_directory, output_base_dir);
            processed_bags++;
        } catch (const std::exception& e) {
            ROS_ERROR("Error processing bag file %s: %s", bag_files[i].c_str(), e.what());
        }
        
        ROS_INFO("");
    }

    ROS_INFO("==========================================");
    ROS_INFO("All processing completed!");
    ROS_INFO("Successfully processed: %d/%d bag files", processed_bags, total_bags);
    ROS_INFO("Output directory: %s", output_base_dir.c_str());
    ROS_INFO("==========================================");
    
    return 0;
}

