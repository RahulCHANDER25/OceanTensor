#include "Linear/LinearNetwork.hpp"
#include "Sequential.hpp"

#include "include/trainings.hpp"

void training()
{
    Network::LinearNetwork net(5, 64);
    Network::LinearNetwork netOut(64, 5);

    Network::Sequential seq{
        std::move(net),
        std::move(netOut)
    };

    fit_training(seq, "test.txt");
}

// ++ Fonction d'activation
// ++ Fonction D'Erreur
// Améliorer l'interface d'utilisation
// +++

int main()
{
    // dot_test();
    // matrix_test();
    // tensor_test();
    // linear_test();
    // test_Sequential();
    // test_save();
    training();
    // copy_constructor_test();
    return 0;
}
