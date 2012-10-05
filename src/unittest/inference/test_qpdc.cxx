#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/inference/qpdc.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

int main() {
   {
      typedef opengm::GraphicalModel<double, opengm::Adder > SumGmType;
      typedef opengm::BlackBoxTestGrid<SumGmType> SumGridTest;
      typedef opengm::BlackBoxTestFull<SumGmType> SumFullTest;
      typedef opengm::BlackBoxTestStar<SumGmType> SumStarTest;

      opengm::InferenceBlackBoxTester<SumGmType> sumTester;    
      sumTester.addTest(new SumStarTest(100,       5, true, true, SumStarTest::RANDOM, opengm::PASS, 1));
      sumTester.addTest(new SumGridTest(20,   10,  5, true, true, SumGridTest::RANDOM, opengm::PASS, 1));
      sumTester.addTest(new SumFullTest(50 ,       3, true,    3, SumFullTest::RANDOM, opengm::PASS, 1));
     
      std::cout << "QPDC Tests"<<std::endl;
      {
         std::cout << "  * Minimization/Adder ..."<<std::endl;
         typedef opengm::QpDC<SumGmType, opengm::Minimizer> QPDC;
         QPDC::Parameter para;
         sumTester.test< QPDC >(para);
         std::cout << " OK!"<<std::endl;
      }
      {
         std::cout << "  * Maximizer/Adder ..."<<std::endl;
         typedef opengm::QpDC<SumGmType, opengm::Maximizer> QPDC;
         QPDC::Parameter para;
         sumTester.test< QPDC >(para);
         std::cout << " OK!"<<std::endl;
      }
      std::cout << "done!"<<std::endl;
   }
};
