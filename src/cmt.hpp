#ifndef CMT_H
#define CMT_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

class CMT {
public:
    CMT(const std::string& detectorType = "Feature2D.BRISK",
            const std::string& descriptorType = "Feature2D.BRISK",
            const std::string& matcherType = "BruteForce-Hamming",
            const int& thrOutlier = 20, const float& thrConf = 0.75,
            const float& thrRatio = 0.8, const int& descriptorLength = 512,
            const bool& estimateScale = true, const bool& estimateRotation =
                    true, const unsigned int& nbInitialKeypoints = 0);
    void initialise(const cv::Mat& im_gray0, const cv::Rect_<float>&);
    void initialise(const cv::Mat& im_gray0, cv::Point2f topleft,
            cv::Point2f bottomright);
    void estimate(const std::vector<std::pair<cv::KeyPoint, int> >& keypointsIN,
            cv::Point2f& center, float& scaleEstimate, float& medRot,
            std::vector<std::pair<cv::KeyPoint, int> >& keypoints);
    void processFrame(cv::Mat im_gray);
private:
    void inout_rect(const std::vector<cv::KeyPoint>& keypoints,
            const cv::Point2f& topleft,
            const cv::Point2f& bottomright,
            std::vector<cv::KeyPoint>& in,
            std::vector<cv::KeyPoint>& out);
    void track(const cv::Mat& im_prev, const cv::Mat& im_gray,
            const std::vector<std::pair<cv::KeyPoint, int>>& keypointsIN,
            std::vector<std::pair<cv::KeyPoint, int>>& keypointsTracked,
            std::vector<unsigned char>& status, int THR_FB = 20);
    cv::Point2f rotate(const cv::Point2f& p, const float& rad);


    std::string m_detectorType;
    std::string m_descriptorType;
    std::string m_matcherType;
    int m_descriptorLength;
    int m_thrOutlier;
    float m_thrConf;
    float m_thrRatio;

    bool m_estimateScale;
    bool m_estimateRotation;

    cv::Ptr<cv::FeatureDetector> m_detector;
    cv::Ptr<cv::DescriptorExtractor> m_descriptorExtractor;
    cv::Ptr<cv::DescriptorMatcher> m_descriptorMatcher;

    cv::Mat m_selectedFeatures;
    std::vector<int> m_selectedClasses;
    cv::Mat m_featuresDatabase;
    std::vector<int> m_classesDatabase;

    std::vector<std::vector<float>> m_squareForm;
    std::vector<std::vector<float>> m_angles;

    cv::Point2f topLeft;
    cv::Point2f topRight;
    cv::Point2f bottomRight;
    cv::Point2f bottomLeft;

    cv::Rect_<float> boundingbox;
    bool hasResult;

    cv::Point2f centerToTopLeft;
    cv::Point2f centerToTopRight;
    cv::Point2f centerToBottomRight;
    cv::Point2f centerToBottomLeft;

    std::vector<cv::Point2f> springs;

    cv::Mat im_prev;
    std::vector<std::pair<cv::KeyPoint, int> > activeKeypoints;
    std::vector<std::pair<cv::KeyPoint, int> > trackedKeypoints;

    unsigned int nbInitialKeypoints;

    std::vector<cv::Point2f> votes;

    std::vector<std::pair<cv::KeyPoint, int> > outliers;
};

class Cluster {
public:
    int first, second; //cluster id
    float dist;
    int num;
};


#endif // CMT_H
