#include <iostream>
#include <fstream>
#include <unordered_map>
#include <sstream>
#include <string>
#include <opencv2/opencv.hpp>

class OneChannelLUT
{
public:
    OneChannelLUT() {
        // create a y = x transform
        lut_mat = cv::Mat(1, 256, CV_8UC1);
        unsigned char* p_data = lut_mat.data;
        for(int i = 0; i < 256; ++i) {
            p_data[i] = i;
        }
    }
    OneChannelLUT(unsigned char* data) {
        lut_mat = cv::Mat(1, 256, CV_8UC1, data);
    }
    OneChannelLUT(const std::string& csv) {
        std::ifstream file(csv);
        assert(file.is_open());
        lut_mat = cv::Mat(1, 256, CV_8UC1);
        unsigned char* p_data = lut_mat.data;
        int val;
        try {
            for(int i = 0; i < 256; ++i) {
                file >> val;
                p_data[i] = val;
            }
        } catch (std::exception& e) {
            std::cout << e.what() << std::endl;
            exit(EXIT_FAILURE);
        }
        file.close();
    }
    int apply_1ch(const cv::Mat& src, cv::Mat& dst) {
        if(src.channels() != 1) {
            std::cout << "src image must be 1 channel" << std::endl;
            return EXIT_FAILURE;
        }
        cv::LUT(src, lut_mat, dst);
        return EXIT_SUCCESS;
    }

    ~OneChannelLUT() {}
public:
    cv::Mat lut_mat;
};

class BrightnessEditor{
public:
    BrightnessEditor() {
        luts.insert({-3, OneChannelLUT("data/beautify_whitening_lut/lookup_table_-3.csv")});
        luts.insert({-2, OneChannelLUT("data/beautify_whitening_lut/lookup_table_-2.csv")});
        luts.insert({-1, OneChannelLUT("data/beautify_whitening_lut/lookup_table_-1.csv")});
        luts.insert({0, OneChannelLUT()});
        luts.insert({1, OneChannelLUT("data/beautify_whitening_lut/lookup_table_+1.csv")});
        luts.insert({2, OneChannelLUT("data/beautify_whitening_lut/lookup_table_+2.csv")});
        luts.insert({3, OneChannelLUT("data/beautify_whitening_lut/lookup_table_+3.csv")});
    }
    ~BrightnessEditor() {}

    int apply(const cv::Mat& src, cv::Mat& dst, float brightness) {
        assert(src.channels() == 3);
        assert(brightness >= -3 && brightness <= 3);
        cv::Mat lut_mat = interpolate(brightness);
        std::cout << lut_mat.type() << " " << lut_mat.cols << "x" << lut_mat.rows << std::endl;

        std::vector<cv::Mat> yuv;
        cv::Mat src_yuv;
        cv::cvtColor(src, src_yuv, cv::COLOR_BGR2YUV);
        cv::split(src_yuv, yuv);

        cv::Mat y_edited;
        cv::LUT(yuv[0], lut_mat, y_edited);

        cv::Mat yuv_edited;
        cv::merge(std::vector<cv::Mat>({y_edited, yuv[1], yuv[2]}), yuv_edited);
        cv::cvtColor(yuv_edited, dst, cv::COLOR_YUV2BGR);

        return EXIT_SUCCESS;
    }

    cv::Mat interpolate(const float val) {
        cv::Mat dst_lut;
        if(val >= -3 && val < -2) {
            cv::addWeighted(luts[-3].lut_mat, 1.0f - (val + 3.0f), luts[-2].lut_mat, val + 3.0f, 0.0f, dst_lut);
        }
        else if(val >= -2 && val < -1) {
            cv::addWeighted(luts[-2].lut_mat, 1.0f - (val + 2.0f), luts[-1].lut_mat, val + 2.0f, 0.0f, dst_lut);
        }
        else if(val >= -1 && val < 0) {
            cv::addWeighted(luts[-1].lut_mat, 1.0f - (val + 1.0f), luts[0].lut_mat, val + 1.0f, 0.0f, dst_lut);
        }
        else if(val >= 0 && val < 1) {
            cv::addWeighted(luts[0].lut_mat, 1.0f - val, luts[1].lut_mat, val, 0.0f, dst_lut);
        }
        else if(val >= 1 && val < 2) {
            cv::addWeighted(luts[1].lut_mat, 1.0f - (val - 1.0f), luts[2].lut_mat, val - 1.0f, 0.0f, dst_lut);
        }
        else if(val >= 2 && val < 3) {
            cv::addWeighted(luts[2].lut_mat, 1.0f - (val - 2.0f), luts[3].lut_mat, val - 2.0f, 0.0f, dst_lut);
        }
        else {
            std::cout << "invalid brightness value" << std::endl;
        }

        return dst_lut;
    }
public:
    std::unordered_map<int, OneChannelLUT> luts;
};

int main() {
    BrightnessEditor be;

    cv::Mat src = cv::imread("./data/shiroi16_fm00017550__0.jpg", cv::IMREAD_COLOR);

    int i=0;
    for(float brightness = -3.0f; brightness <= 3.0f; brightness += 0.1f, ++i) {
        std::ostringstream oss;
        oss << "brightness_" << std::setw(2) << std::setfill('0') << i << "_" << std::setprecision(2) << brightness << ".jpg";
        std::cout << "saved " << oss.str() << std::endl;
        cv::Mat dst;
        be.apply(src, dst, brightness);
        cv::imwrite(oss.str().c_str(), dst);
    }

    return EXIT_SUCCESS;
}
