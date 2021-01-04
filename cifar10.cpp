#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iostream>
#define ATD at<float>
#define elif else if

using namespace cv;
using namespace std;

void read_batch(string filename, vector<Mat> &vec, Mat &label) {
    ifstream file(filename, ios::binary);
    for (int i = 0; i < 10000; ++i) {
      unsigned char tplabel = 0;
      file.read((char*)&tplabel, sizeof(tplabel));

      char *data = new char[3072];
      file.read(data, 3072);
      Mat im(32, 32, CV_8UC3);
      for (int r = 0; r < im.rows; r++) {
        for (int c = 0; c < im.cols; c++) {
          im.at<Vec3b>(r, c) = Vec3b(data[c + r * 32],
                                     data[c + r * 32 + 32 * 32],
                                     data[c + r * 32 + 32 * 32 * 2]);
        }
      }
      vec.push_back(im);
      label.ATD(0, i) = (float) tplabel;
    }
}

Mat concatenateMat(vector<Mat> &vec) {
    int height = vec[0].rows;
    int width = vec[0].cols;
    Mat res = Mat::zeros(height * width, vec.size(), CV_32FC1);
    for (unsigned i = 0; i < vec.size(); i++) {
        Mat img(height, width, CV_32FC1);
        Mat gray(height, width, CV_8UC1);
        cvtColor(vec[i], gray, CV_RGB2GRAY);
        gray.convertTo(img, CV_32FC1);
        // reshape(int cn, int rows=0), cn is num of channels.
        Mat ptmat = img.reshape(0, height * width);
        Rect roi = cv::Rect(i, 0, ptmat.cols, ptmat.rows);
        Mat subView = res(roi);
        ptmat.copyTo(subView);
    }
    divide(res, 255.0, res);
    return res;
}

void read_CIFAR10(Mat &trainX, Mat &testX, const Mat &trainY, Mat &testY) {
    string filename;
    filename = "cifar-10-batches-bin/data_batch_1.bin";
    vector<Mat> batch1;
    Mat label1 = Mat::zeros(1, 10000, CV_32FC1);
    read_batch(filename, batch1, label1);

    filename = "cifar-10-batches-bin/data_batch_2.bin";
    vector<Mat> batch2;
    Mat label2 = Mat::zeros(1, 10000, CV_32FC1);
    read_batch(filename, batch2, label2);

    filename = "cifar-10-batches-bin/data_batch_3.bin";
    vector<Mat> batch3;
    Mat label3 = Mat::zeros(1, 10000, CV_32FC1);
    read_batch(filename, batch3, label3);

    filename = "cifar-10-batches-bin/data_batch_4.bin";
    vector<Mat> batch4;
    Mat label4 = Mat::zeros(1, 10000, CV_32FC1);
    read_batch(filename, batch4, label4);

    filename = "cifar-10-batches-bin/data_batch_5.bin";
    vector<Mat> batch5;
    Mat label5 = Mat::zeros(1, 10000, CV_32FC1);
    read_batch(filename, batch5, label5);

    filename = "cifar-10-batches-bin/test_batch.bin";
    vector<Mat> batcht;
    Mat labelt = Mat::zeros(1, 10000, CV_32FC1);
    read_batch(filename, batcht, labelt);

    Mat mt1 = concatenateMat(batch1);
    Mat mt2 = concatenateMat(batch2);
    Mat mt3 = concatenateMat(batch3);
    Mat mt4 = concatenateMat(batch4);
    Mat mt5 = concatenateMat(batch5);
    Mat mtt = concatenateMat(batcht);

    Rect roi = cv::Rect(mt1.cols * 0, 0, mt1.cols, trainX.rows);
    Mat subView = trainX(roi);
    mt1.copyTo(subView);

    roi = cv::Rect(label1.cols * 0, 0, label1.cols, 1);
    subView = trainY(roi);
    label1.copyTo(subView);

    roi = cv::Rect(mt1.cols * 1, 0, mt1.cols, trainX.rows);
    subView = trainX(roi);
    mt2.copyTo(subView);

    roi = cv::Rect(label1.cols * 1, 0, label1.cols, 1);
    subView = trainY(roi);
    label2.copyTo(subView);

    roi = cv::Rect(mt1.cols * 2, 0, mt1.cols, trainX.rows);
    subView = trainX(roi);
    mt3.copyTo(subView);

    roi = cv::Rect(label1.cols * 2, 0, label1.cols, 1);
    subView = trainY(roi);
    label3.copyTo(subView);

    roi = cv::Rect(mt1.cols * 3, 0, mt1.cols, trainX.rows);
    subView = trainX(roi);
    mt4.copyTo(subView);

    roi = cv::Rect(label1.cols * 3, 0, label1.cols, 1);
    subView = trainY(roi);
    label4.copyTo(subView);

    roi = cv::Rect(mt1.cols * 4, 0, mt1.cols, trainX.rows);
    subView = trainX(roi);
    mt5.copyTo(subView);

    roi = cv::Rect(label1.cols * 4, 0, label1.cols, 1);
    subView = trainY(roi);
    label5.copyTo(subView);

    mtt.copyTo(testX);
    labelt.copyTo(testY);
}
