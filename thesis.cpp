#include "opencl.hpp"
#include "convnn.hpp"
#include "kernel.c"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

void read_Mnist(std::string filename, std::vector<std::vector<float>> &vec);
void read_Mnist_Label(std::string filename, std::vector<std::vector<float>> &vec, std::vector<float> &testtargets, bool testflag);
void printInput(std::vector<float> &inputs);
void read_CIFAR10(cv::Mat &trainX, cv::Mat &testX, const cv::Mat &trainY, cv::Mat &testY);

int main(void)
{
  try {
    OpenCL::initialize_OpenCL();

    ///CIFAR10
    /////////////////////////////////////////////////////////
    std::clock_t start = std::clock();
    cv::Mat trainX, testX;
    cv::Mat trainY, testY;
    trainX = cv::Mat::zeros(1024, 50000, CV_32FC1);
    testX = cv::Mat::zeros(1024, 10000, CV_32FC1);
    trainY = cv::Mat::zeros(1, 50000, CV_32FC1);
    testY = cv::Mat::zeros(1, 10000, CV_32FC1);
    read_CIFAR10(trainX, testX, trainY, testY);
    std::cout << "Cifar10 loaded in: " << (std::clock() - start) / (double) CLOCKS_PER_SEC << " s" << std::endl;

    start = std::clock();
    std::vector<std::vector<float> > inputs(50000), targets(50000);
    for (int i = 0; i < 50000; i++) {
      inputs.push_back(trainX.col(i));
      std::vector<float> tempvec(10);
      for (int j = 0; j < 10; j++) {
        tempvec[j] = j == trainY.col(i).at<float>(0) ? 1 : 0;
      }
      targets.push_back(tempvec);
    }
    std::vector<std::vector<float> > testinputs;
    std::vector<float> testtargets;
    for (int i = 0; i < 10000; i++) {
      testinputs.push_back(testX.col(i));
      testtargets.push_back(testY.col(i).at<float>(0));
    }
    std::cout << "Cifar10 converted in: " << (std::clock() - start) / (double) CLOCKS_PER_SEC << " s" << std::endl;

    start = std::clock();
    ConvNN m_nn;
    m_nn.createConvNN(7, 7, 32);//num of filters,filterdim,imagedim
    //todo::many filters  3d kernel
    std::vector<int> netVec;
    netVec = { 169 * 7,10 };
    m_nn.createFullyConnectedNN(netVec, 0, 32);
    m_nn.train(inputs, targets, testinputs, testtargets, 1000000);
    std::cout << "trained in : " << (std::clock() - start) / (double) CLOCKS_PER_SEC << " s" << std::endl;

    ///////////////////////////////////////////////////
    /// FCNN
    ///////////////////////////////////////////////////

    ConvNN f_nn;
    std::vector<int> fnetVec({ 48,10 });
    f_nn.createFullyConnectedNN(fnetVec, 1, 32);
    f_nn.forwardFCNN(inputs[0]);
    f_nn.trainFCNN(inputs, targets, testinputs, testtargets, 50000);
    std::cout << "trained in : " << (std::clock() - start) / (double) CLOCKS_PER_SEC << " s" << std::endl;
    f_nn.trainingAccuracy(testinputs, testtargets, 2000, 1);
  } catch (const cl::Error *e) {
    std::cout << "opencl error: " << e->what() << std::endl;
    std::cout << "error number: " << e->err() << std::endl;
  } catch (const int e) {
    std::cout << "An exception occurred. Exception Nr. " << e << '\n';
  }
}
