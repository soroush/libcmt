#include "cmt.hpp"
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <cmath>

#if __cplusplus < 201103L //test if c++11

#include <limits>

#ifndef NAN
//may not be correct on all compilator, DON'T USE the flag FFAST-MATH

#define NAN numeric_limits<float>::quiet_NaN()

template <T>
bool isnan(T d)
{
    return d != d;
}
#endif

#endif

using namespace std;
using namespace cv;

void inout_rect(const vector<KeyPoint>& keypoints, Point2f topleft,
        Point2f bottomright, vector<KeyPoint>& in, vector<KeyPoint>& out) {
    for (unsigned int i = 0; i < keypoints.size(); i++) {
        if (keypoints[i].pt.x > topleft.x && keypoints[i].pt.y > topleft.y
                && keypoints[i].pt.x < bottomright.x
                && keypoints[i].pt.y < bottomright.y)
            in.push_back(keypoints[i]);
        else
            out.push_back(keypoints[i]);
    }
}

void track(Mat im_prev, Mat im_gray,
        const vector<pair<KeyPoint, int> >& keypointsIN,
        vector<pair<KeyPoint, int> >& keypointsTracked,
        vector<unsigned char>& status, int THR_FB) {
    //Status of tracked keypoint - True means successfully tracked
    status = vector<unsigned char>();
    //for(int i = 0; i < keypointsIN.size(); i++)
    //  status.push_back(false);
    //If at least one keypoint is active
    if (keypointsIN.size() > 0) {
        vector<Point2f> pts;
        vector<Point2f> pts_back;
        vector<Point2f> nextPts;
        vector<unsigned char> status_back;
        vector<float> err;
        vector<float> err_back;
        vector<float> fb_err;
        for (unsigned int i = 0; i < keypointsIN.size(); i++)
            pts.push_back(
                    Point2f(keypointsIN[i].first.pt.x,
                            keypointsIN[i].first.pt.y));

        //Calculate forward optical flow for prev_location
        calcOpticalFlowPyrLK(im_prev, im_gray, pts, nextPts, status, err);
        //Calculate backward optical flow for prev_location
        calcOpticalFlowPyrLK(im_gray, im_prev, nextPts, pts_back, status_back,
                err_back);

        //Calculate forward-backward error
        for (unsigned int i = 0; i < pts.size(); i++) {
            Point2f v = pts_back[i] - pts[i];
            fb_err.push_back(sqrt(v.dot(v)));
        }

        //Set status depending on fb_err and lk error
        for (unsigned int i = 0; i < status.size(); i++)
            status[i] = (fb_err[i] <= THR_FB) & status[i];

        keypointsTracked = vector<pair<KeyPoint, int> >();
        for (unsigned int i = 0; i < pts.size(); i++) {
            pair<KeyPoint, int> p = keypointsIN[i];
            if (status[i]) {
                p.first.pt = nextPts[i];
                keypointsTracked.push_back(p);
            }
        }
    }
    else
        keypointsTracked = vector<pair<KeyPoint, int> >();
}

Point2f rotate(Point2f p, float rad) {
    if (rad == 0)
        return p;
    float s = sin(rad);
    float c = cos(rad);
    return Point2f(c * p.x - s * p.y, s * p.x + c * p.y);
}

CMT::CMT(const string& detectorType_, const string& descriptorType_,
        const string& matcherType_, const int& thrOutlier_,
        const float& thrConf_, const float& thrRatio_,
        const int& descriptorLength_, const bool& estimateScale_,
        const bool& estimateRotation_, const unsigned int& nbInitialKeypoints_) :
        m_detectorType { detectorType_ },
        m_descriptorType { descriptorType_ },
        m_matcherType { matcherType_ },
        m_thrOutlier { thrOutlier_ },
        m_thrConf { thrConf_ },
        m_thrRatio { thrRatio_ },
        m_descriptorLength { descriptorLength_ },
        m_estimateScale { estimateScale_ },
        m_estimateRotation { estimateRotation_ },
        nbInitialKeypoints { nbInitialKeypoints_ } {
}

void CMT::initialise(const Mat& im_gray0, const Rect_<float>& rect) {
    initialise(im_gray0, rect.tl(), rect.br());
}

void CMT::initialise(const Mat& im_gray0, Point2f topleft,
        Point2f bottomright) {
    //Initialise detector, descriptor, matcher
    m_detector = Algorithm::create<FeatureDetector>(m_detectorType.c_str());
    m_descriptorExtractor = Algorithm::create<DescriptorExtractor>(
            m_descriptorType.c_str());
    m_descriptorMatcher = DescriptorMatcher::create(m_matcherType.c_str());
    vector<string> list;
    Algorithm::getList(list);

    //Get initial keypoints in whole image
    vector<KeyPoint> keypoints;
    m_detector->detect(im_gray0, keypoints);

    //Remember keypoints that are in the rectangle as selected keypoints
    vector<KeyPoint> selected_keypoints;
    vector<KeyPoint> background_keypoints;
    inout_rect(keypoints, topleft, bottomright, selected_keypoints,
            background_keypoints);
    m_descriptorExtractor->compute(im_gray0, selected_keypoints,
            m_selectedFeatures);

    if (selected_keypoints.size() == 0) {
        printf("No keypoints found in selection");
        return;
    }

    //Remember keypoints that are not in the rectangle as background keypoints
    Mat background_features;
    m_descriptorExtractor->compute(im_gray0, background_keypoints,
            background_features);

    //Assign each keypoint a class starting from 1, background is 0
    m_selectedClasses = vector<int>();
    for (unsigned int i = 1; i <= selected_keypoints.size(); i++)
        m_selectedClasses.push_back(i);
    vector<int> backgroundClasses;
    for (unsigned int i = 0; i < background_keypoints.size(); i++)
        backgroundClasses.push_back(0);

    //Stack background features and selected features into database
    m_featuresDatabase = Mat(background_features.rows + m_selectedFeatures.rows,
            max(background_features.cols, m_selectedFeatures.cols),
            background_features.type());
    if (background_features.cols > 0)
        background_features.copyTo(
                m_featuresDatabase(
                        Rect(0, 0, background_features.cols,
                                background_features.rows)));
    if (m_selectedFeatures.cols > 0)
        m_selectedFeatures.copyTo(
                m_featuresDatabase(
                        Rect(0, background_features.rows,
                                m_selectedFeatures.cols,
                                m_selectedFeatures.rows)));

    //Same for classes
    m_classesDatabase = vector<int>();
    for (unsigned int i = 0; i < backgroundClasses.size(); i++)
        m_classesDatabase.push_back(backgroundClasses[i]);
    for (unsigned int i = 0; i < m_selectedClasses.size(); i++)
        m_classesDatabase.push_back(m_selectedClasses[i]);

    //Get all distances between selected keypoints in squareform and get all angles between selected keypoints
    m_squareForm = vector<vector<float> >();
    m_angles = vector<vector<float> >();
    for (unsigned int i = 0; i < selected_keypoints.size(); i++) {
        vector<float> lineSquare;
        vector<float> lineAngle;
        for (unsigned int j = 0; j < selected_keypoints.size(); j++) {
            float dx = selected_keypoints[j].pt.x - selected_keypoints[i].pt.x;
            float dy = selected_keypoints[j].pt.y - selected_keypoints[i].pt.y;
            lineSquare.push_back(sqrt(dx * dx + dy * dy));
            lineAngle.push_back(atan2(dy, dx));
        }
        m_squareForm.push_back(lineSquare);
        m_angles.push_back(lineAngle);
    }

    //Find the center of selected keypoints
    Point2f center(0, 0);
    for (unsigned int i = 0; i < selected_keypoints.size(); i++)
        center += selected_keypoints[i].pt;
    center *= (1.0 / selected_keypoints.size());

    //Remember the rectangle coordinates relative to the center
    centerToTopLeft = topleft - center;
    centerToTopRight = Point2f(bottomright.x, topleft.y) - center;
    centerToBottomRight = bottomright - center;
    centerToBottomLeft = Point2f(topleft.x, bottomright.y) - center;

    //Calculate springs of each keypoint
    springs = vector<Point2f>();
    for (unsigned int i = 0; i < selected_keypoints.size(); i++)
        springs.push_back(selected_keypoints[i].pt - center);

    //Set start image for tracking
    im_prev = im_gray0.clone();

    //Make keypoints 'active' keypoints
    activeKeypoints = vector<pair<KeyPoint, int> >();
    for (unsigned int i = 0; i < selected_keypoints.size(); i++)
        activeKeypoints.push_back(
                make_pair(selected_keypoints[i], m_selectedClasses[i]));

    //Remember number of initial keypoints
    nbInitialKeypoints = selected_keypoints.size();
}

typedef pair<int, int> PairInt;
typedef pair<float, int> PairFloat;

template<typename T>
bool comparatorPair(const pair<T, int>& l, const pair<T, int>& r) {
    return l.first < r.first;
}

template<typename T>
bool comparatorPairDesc(const pair<T, int>& l, const pair<T, int>& r) {
    return l.first > r.first;
}

template<typename T>
T sign(T t) {
    if (t == 0)
        return T(0);
    else
        return (t < 0) ? T(-1) : T(1);
}

template<typename T>
T median(vector<T> list) {
    T val;
    nth_element(&list[0], &list[0] + list.size() / 2, &list[0] + list.size());
    val = list[list.size() / 2];
    if (list.size() % 2 == 0) {
        nth_element(&list[0], &list[0] + list.size() / 2 - 1,
                &list[0] + list.size());
        val = (val + list[list.size() / 2 - 1]) / 2;
    }
    return val;
}

float findMinSymetric(const vector<vector<float> >& dist,
        const vector<bool>& used, int limit, int &i, int &j) {
    float min = dist[0][0];
    i = 0;
    j = 0;
    for (int x = 0; x < limit; x++) {
        if (!used[x]) {
            for (int y = x + 1; y < limit; y++)
                if (!used[y] && dist[x][y] <= min) {
                    min = dist[x][y];
                    i = x;
                    j = y;
                }
        }
    }
    return min;
}

vector<Cluster> linkage(const vector<Point2f>& list) {
    float inf = 10000000.0;
    vector<bool> used;
    for (unsigned int i = 0; i < 2 * list.size(); i++)
        used.push_back(false);
    vector<vector<float> > dist;
    for (unsigned int i = 0; i < list.size(); i++) {
        vector<float> line;
        for (unsigned int j = 0; j < list.size(); j++) {
            if (i == j)
                line.push_back(inf);
            else {
                Point2f p = list[i] - list[j];
                line.push_back(sqrt(p.dot(p)));
            }
        }
        for (unsigned int j = 0; j < list.size(); j++)
            line.push_back(inf);
        dist.push_back(line);
    }
    for (unsigned int i = 0; i < list.size(); i++) {
        vector<float> line;
        for (unsigned int j = 0; j < 2 * list.size(); j++)
            line.push_back(inf);
        dist.push_back(line);
    }
    vector<Cluster> clusters;
    while (clusters.size() < list.size() - 1) {
        int x, y;
        float minValue = findMinSymetric(dist, used, list.size() + clusters.size(),
                x, y);
        Cluster cluster;
        cluster.first = x;
        cluster.second = y;
        cluster.dist = minValue;
        cluster.num =
                (x < (int) list.size() ? 1 : clusters[x - list.size()].num)
                        + (y < (int) list.size() ?
                                1 : clusters[y - list.size()].num);
        used[x] = true;
        used[y] = true;
        int limit = list.size() + clusters.size();
        for (int i = 0; i < limit; i++) {
            if (!used[i])
                dist[i][limit] = dist[limit][i] = min(dist[i][x], dist[i][y]);
        }
        clusters.push_back(cluster);
    }
    return clusters;
}

void fcluster_rec(vector<int>& data, const vector<Cluster>& clusters,
        float threshold, const Cluster& currentCluster, int& binId) {
    int startBin = binId;
    if (currentCluster.first >= (int) data.size())
        fcluster_rec(data, clusters, threshold,
                clusters[currentCluster.first - data.size()], binId);
    else
        data[currentCluster.first] = binId;

    if (startBin == binId && currentCluster.dist >= threshold)
        binId++;
    startBin = binId;

    if (currentCluster.second >= (int) data.size())
        fcluster_rec(data, clusters, threshold,
                clusters[currentCluster.second - data.size()], binId);
    else
        data[currentCluster.second] = binId;

    if (startBin == binId && currentCluster.dist >= threshold)
        binId++;
}

vector<int> binCount(const vector<int>& T) {
    vector<int> result;
    for (unsigned int i = 0; i < T.size(); i++) {
        while (T[i] >= (int) result.size())
            result.push_back(0);
        result[T[i]]++;
    }
    return result;
}

int argmax(const vector<int>& list) {
    int max = list[0];
    int id = 0;
    for (unsigned int i = 1; i < list.size(); i++)
        if (list[i] > max) {
            max = list[i];
            id = i;
        }
    return id;
}

vector<int> fcluster(const vector<Cluster>& clusters, float threshold) {
    vector<int> data;
    for (unsigned int i = 0; i < clusters.size() + 1; i++)
        data.push_back(0);
    int binId = 0;
    fcluster_rec(data, clusters, threshold, clusters[clusters.size() - 1],
            binId);
    return data;
}

void CMT::estimate(const vector<pair<KeyPoint, int> >& keypointsIN,
        Point2f& center, float& scaleEstimate, float& medRot,
        vector<pair<KeyPoint, int> >& keypoints) {
    center = Point2f(NAN, NAN);
    scaleEstimate = NAN;
    medRot = NAN;

    //At least 2 keypoints are needed for scale
    if (keypointsIN.size() > 1) {
        //sort
        vector<PairInt> list;
        for (unsigned int i = 0; i < keypointsIN.size(); i++)
            list.push_back(make_pair(keypointsIN[i].second, i));
        sort(&list[0], &list[0] + list.size(), comparatorPair<int>);
        for (unsigned int i = 0; i < list.size(); i++)
            keypoints.push_back(keypointsIN[list[i].second]);

        vector<int> ind1;
        vector<int> ind2;
        for (unsigned int i = 0; i < list.size(); i++)
            for (unsigned int j = 0; j < list.size(); j++) {
                if (i != j && keypoints[i].second != keypoints[j].second) {
                    ind1.push_back(i);
                    ind2.push_back(j);
                }
            }
        if (ind1.size() > 0) {
            vector<int> class_ind1;
            vector<int> class_ind2;
            vector<KeyPoint> pts_ind1;
            vector<KeyPoint> pts_ind2;
            for (unsigned int i = 0; i < ind1.size(); i++) {
                class_ind1.push_back(keypoints[ind1[i]].second - 1);
                class_ind2.push_back(keypoints[ind2[i]].second - 1);
                pts_ind1.push_back(keypoints[ind1[i]].first);
                pts_ind2.push_back(keypoints[ind2[i]].first);
            }

            vector<float> scaleChange;
            vector<float> angleDiffs;
            for (unsigned int i = 0; i < pts_ind1.size(); i++) {
                Point2f p = pts_ind2[i].pt - pts_ind1[i].pt;
                //This distance might be 0 for some combinations,
                //as it can happen that there is more than one keypoint at a single location
                float dist = sqrt(p.dot(p));
                float origDist = m_squareForm[class_ind1[i]][class_ind2[i]];
                scaleChange.push_back(dist / origDist);
                //Compute angle
                float angle = atan2(p.y, p.x);
                float origAngle = m_angles[class_ind1[i]][class_ind2[i]];
                float angleDiff = angle - origAngle;
                //Fix long way angles
                if (fabs(angleDiff) > CV_PI)
                    angleDiff -= sign(angleDiff) * 2 * CV_PI;
                angleDiffs.push_back(angleDiff);
            }
            scaleEstimate = median(scaleChange);
            if (!m_estimateScale)
                scaleEstimate = 1;
            medRot = median(angleDiffs);
            if (!m_estimateRotation)
                medRot = 0;
            votes = vector<Point2f>();
            for (unsigned int i = 0; i < keypoints.size(); i++)
                votes.push_back(
                        keypoints[i].first.pt
                                - scaleEstimate
                                        * rotate(
                                                springs[keypoints[i].second - 1],
                                                medRot));
            //Compute linkage between pairwise distances
            vector<Cluster> linkageData = linkage(votes);

            //Perform hierarchical distance-based clustering
            vector<int> T = fcluster(linkageData, m_thrOutlier);
            //Count votes for each cluster
            vector<int> cnt = binCount(T);
            //Get largest class
            int Cmax = argmax(cnt);

            //Remember outliers
            outliers = vector<pair<KeyPoint, int> >();
            vector<pair<KeyPoint, int> > newKeypoints;
            vector<Point2f> newVotes;
            for (unsigned int i = 0; i < keypoints.size(); i++) {
                if (T[i] != Cmax)
                    outliers.push_back(keypoints[i]);
                else {
                    newKeypoints.push_back(keypoints[i]);
                    newVotes.push_back(votes[i]);
                }
            }
            keypoints = newKeypoints;

            center = Point2f(0, 0);
            for (unsigned int i = 0; i < newVotes.size(); i++)
                center += newVotes[i];
            center *= (1.0 / newVotes.size());
        }
    }
}

//todo : n*log(n) by sorting the second array and dichotomic search instead of n^2
vector<bool> in1d(const vector<int>& a, const vector<int>& b) {
    vector<bool> result;
    for (unsigned int i = 0; i < a.size(); i++) {
        bool found = false;
        for (unsigned int j = 0; j < b.size(); j++)
            if (a[i] == b[j]) {
                found = true;
                break;
            }
        result.push_back(found);
    }
    return result;
}

void CMT::processFrame(Mat im_gray) {
    trackedKeypoints = vector<pair<KeyPoint, int> >();
    vector<unsigned char> status;
    track(im_prev, im_gray, activeKeypoints, trackedKeypoints, status);

    Point2f center;
    float scaleEstimate;
    float rotationEstimate;
    vector<pair<KeyPoint, int> > trackedKeypoints2;
    estimate(trackedKeypoints, center, scaleEstimate, rotationEstimate,
            trackedKeypoints2);
    trackedKeypoints = trackedKeypoints2;

    //Detect keypoints, compute descriptors
    vector<KeyPoint> keypoints;
    Mat features;
    m_detector->detect(im_gray, keypoints);
    m_descriptorExtractor->compute(im_gray, keypoints, features);

    //Create list of active keypoints
    activeKeypoints = vector<pair<KeyPoint, int> >();

    //For each keypoint and its descriptor
    for (unsigned int i = 0; i < keypoints.size(); i++) {
        KeyPoint keypoint = keypoints[i];

        //First: Match over whole image
        //Compute distances to all descriptors
        vector<DMatch> matches;
        m_descriptorMatcher->match(m_featuresDatabase, features.row(i),
                matches);

        //Convert distances to confidences, do not weight
        vector<float> combined;
        for (unsigned int j = 0; j < matches.size(); j++)
            combined.push_back(1 - matches[j].distance / m_descriptorLength);

        vector<int>& classes = m_classesDatabase;

        //Sort in descending order
        vector<PairFloat> sorted_conf;
        for (unsigned int j = 0; j < combined.size(); j++)
            sorted_conf.push_back(make_pair(combined[j], j));
        sort(&sorted_conf[0], &sorted_conf[0] + sorted_conf.size(),
                comparatorPairDesc<float>);

        // Get best and second best index
        int bestInd = sorted_conf[0].second;
        int secondBestInd = sorted_conf[1].second;

        // Compute distance ratio according to Lowe
        float ratio;
        if (secondBestInd < combined.size() and bestInd < combined.size()) {
            if (1 - combined[secondBestInd] != 0) {
                ratio = (1 - combined[bestInd]) / (1 - combined[secondBestInd]);
            }
            else {
                ratio = numeric_limits<float>::max();
            }
        }
        else {
            ratio = numeric_limits<float>::max();
        }

        //Extract class of best match
        int keypoint_class = classes[bestInd];

        //If distance ratio is ok and absolute distance is ok and keypoint class is not background
        if (ratio < m_thrRatio && combined[bestInd] > m_thrConf
                && keypoint_class != 0)
            activeKeypoints.push_back(make_pair(keypoint, keypoint_class));

        //In a second step, try to match difficult keypoints
        //If structural constraints are applicable
        if (!(isnan(center.x) | isnan(center.y))) {
            //Compute distances to initial descriptors
            vector<DMatch> matches;
            m_descriptorMatcher->match(m_selectedFeatures, features.row(i),
                    matches);

            //Convert distances to confidences
            vector<float> confidences;
            for (unsigned int i = 0; i < matches.size(); i++)
                confidences.push_back(
                        1 - matches[i].distance / m_descriptorLength);

            //Compute the keypoint location relative to the object center
            Point2f relative_location = keypoint.pt - center;

            //Compute the distances to all springs
            vector<float> displacements;
            for (unsigned int i = 0; i < springs.size(); i++) {
                Point2f p = (scaleEstimate
                        * rotate(springs[i], -rotationEstimate)
                        - relative_location);
                displacements.push_back(sqrt(p.dot(p)));
            }

            //For each spring, calculate weight
            vector<float> combined;
            for (unsigned int i = 0; i < confidences.size(); i++)
                combined.push_back(
                        (displacements[i] < m_thrOutlier) * confidences[i]);

            vector<int>& classes = m_selectedClasses;

            //Sort in descending order
            vector<PairFloat> sorted_conf;
            for (unsigned int i = 0; i < combined.size(); i++)
                sorted_conf.push_back(make_pair(combined[i], i));
            sort(&sorted_conf[0], &sorted_conf[0] + sorted_conf.size(),
                    comparatorPairDesc<float>);

            //Get best and second best index
            int bestInd = sorted_conf[0].second;
            int secondBestInd = sorted_conf[1].second;

            //Compute distance ratio according to Lowe
            float ratio = (1 - combined[bestInd])
                    / (1 - combined[secondBestInd]);

            //Extract class of best match
            int keypoint_class = classes[bestInd];

            //If distance ratio is ok and absolute distance is ok and keypoint class is not background
            if (ratio < m_thrRatio && combined[bestInd] > m_thrConf
                    && keypoint_class != 0) {
                for (int i = activeKeypoints.size() - 1; i >= 0; i--)
                    if (activeKeypoints[i].second == keypoint_class)
                        activeKeypoints.erase(activeKeypoints.begin() + i);
                activeKeypoints.push_back(make_pair(keypoint, keypoint_class));
            }
        }
    }

    //If some keypoints have been tracked
    if (trackedKeypoints.size() > 0) {
        //Extract the keypoint classes
        vector<int> tracked_classes;
        for (unsigned int i = 0; i < trackedKeypoints.size(); i++)
            tracked_classes.push_back(trackedKeypoints[i].second);
        //If there already are some active keypoints
        if (activeKeypoints.size() > 0) {
            //Add all tracked keypoints that have not been matched
            vector<int> associated_classes;
            for (unsigned int i = 0; i < activeKeypoints.size(); i++)
                associated_classes.push_back(activeKeypoints[i].second);
            vector<bool> notmissing = in1d(tracked_classes, associated_classes);
            for (unsigned int i = 0; i < trackedKeypoints.size(); i++)
                if (!notmissing[i])
                    activeKeypoints.push_back(trackedKeypoints[i]);
        }
        else
            activeKeypoints = trackedKeypoints;
    }

    //Update object state estimate
    vector<pair<KeyPoint, int> > activeKeypointsBefore = activeKeypoints;
    im_prev = im_gray;
    topLeft = Point2f(NAN, NAN);
    topRight = Point2f(NAN, NAN);
    bottomLeft = Point2f(NAN, NAN);
    bottomRight = Point2f(NAN, NAN);

    boundingbox = Rect_<float>(NAN, NAN, NAN, NAN);
    hasResult = false;
    if (!(isnan(center.x) | isnan(center.y))
            && activeKeypoints.size() > nbInitialKeypoints / 10) {
        hasResult = true;

        topLeft = center
                + scaleEstimate * rotate(centerToTopLeft, rotationEstimate);
        topRight = center
                + scaleEstimate * rotate(centerToTopRight, rotationEstimate);
        bottomLeft = center
                + scaleEstimate * rotate(centerToBottomLeft, rotationEstimate);
        bottomRight = center
                + scaleEstimate * rotate(centerToBottomRight, rotationEstimate);

        float minx = min(min(topLeft.x, topRight.x),
                min(bottomRight.x, bottomLeft.x));
        float miny = min(min(topLeft.y, topRight.y),
                min(bottomRight.y, bottomLeft.y));
        float maxx = max(max(topLeft.x, topRight.x),
                max(bottomRight.x, bottomLeft.x));
        float maxy = max(max(topLeft.y, topRight.y),
                max(bottomRight.y, bottomLeft.y));

        boundingbox = Rect_<float>(minx, miny, maxx - minx, maxy - miny);
    }
}
