#include "Linear/LinearNetwork.hpp"
#include "Sequential.hpp"

#include "include/trainings.hpp"
#include <fstream>

void training(
    const size_t EPOCH,
    const std::string &saveName
)
{
    Network::LinearNetwork net(5, 64);
    Network::LinearNetwork netOut(64, 5);

    Network::Sequential seq{
        std::move(net),
        std::move(netOut)
    };

    func_predict_training(seq, EPOCH, saveName);
}

void testing_load(const std::string &name)
{
    Network::Sequential seq;
    std::ifstream ifs(name);

    // Put function sigmoid to them or store it.
    seq.load(ifs);
    std::cout << "Load !" << std::endl;
    std::cout << seq.in() << std::endl;
    std::cout << seq.out() << std::endl;
    Matrix2f input({5, 1});
    Matrix2f y({5, 1});
    for (int i = 0; i < input.size(); i++) {
        y[i] = cos(input[i]);
    }
    Matrix2f pred = seq.forward(input);
    for (int i = 0; i < y.size(); i++)
        std::cout << "Y = " << y[i] << "\tPRED = " << pred[i] << std::endl;
}

void testing_lin_load()
{
    Network::LinearNetwork lin({5, 1});
    Network::LinearNetwork l;

    lin.save("lin.txt", std::ios::trunc);
    std::cout << lin.m_weight << std::endl;
    std::cout << lin.m_bias << std::endl;
    std::ifstream ifs("lin.txt");

    l.load(ifs);
    std::cout << l.m_weight << std::endl;
    std::cout << l.m_bias << std::endl;
}

// ---> Tester sur un linear le bias !
// ++ Fonction d'activation
// ++ Fonction D'Erreur
// Améliorer l'interface d'utilisation
// +++

int main()
{
    training(10, "Rahul.txt");
    // testing_load("test.txt");
    // testing_lin_load();
    return 0;
}
