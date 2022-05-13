#ifndef BASISFUNCTION_H
#define BASISFUNCTION_H

#include <cmath>
#include <vector>
#include <stdexcept>
#include <armadillo>
#include <string>

using namespace std;

class Atom;

class BasisFunction {
 public:
    void calculate_normalizations(); /* Calculate normalization constants for primitive Gaussians */
    //Default constructor
    BasisFunction() : m_name("default"), m_atom(nullptr), m_R(arma::vec(3, arma::fill::zeros)), m_momentum(arma::vec({0., 0., 0.})), m_alphas(arma::vec(3, arma::fill::ones)),  m_contractions(arma::vec(3, arma::fill::ones)),  m_normalizations(arma::vec(3, arma::fill::ones)) {calculate_normalizations();}
    //Constructor
    BasisFunction(string name, const arma::vec& R, const arma::vec& momentum, const arma::vec& alphas, const arma::vec& contractions) : m_name(name) {
        //Center should be of 3 dimensions
        if (R.size() != 3) {
            throw std::invalid_argument("R must be a vector of size 3 containing the x, y, and z coordinates of the shell center.");
        }

        // Momentum should be of 3 dimensions (l, m, n)
        if (momentum.size() !=3) {
            throw std::invalid_argument("momentum must be a vector of size 3 containing the l, m, and n values of the shell");
        }

        //Alphas and Contraction coefficients should be of 3 dimensions
        if (alphas.size() !=3 || contractions.size() != 3) {
            throw std::invalid_argument("alphas and contractions must be vectors of size 3 containing the alpha or contraction coefficient values of the each of the primitive functions");
        }

        m_R = R;
        m_momentum = momentum;
        m_alphas = alphas;
        m_contractions = contractions;

        calculate_normalizations();
    }

    void set_atom(Atom* atom) {
        if (atom != nullptr)
            m_atom = atom;
    }

    void print_info() {
        /* Function to print information about the basis function
         */
        cout << "This AO's info: " << m_name << endl;
        m_R.print("R");
        m_momentum.print("angular momentum");
        m_alphas.print("alphas");
        m_contractions.print("contraction coefficients");
        m_normalizations.print("normalization constants");
    }

    void set_R(const arma::vec& R) {
        if (R.size() != 3) {
            throw std::invalid_argument("R must be a vector of size 3 containing the x, y, and z coordinates of the shell center.");
        }
        m_R = R;
    }

    string get_name() const {
        /* Getter function for orbital name
         */
        return m_name;
    }

    Atom* get_atom() const {
        /* Getter function for orbital name
         */
        return m_atom;
    }

    arma::vec get_R() const {
        /* Getter function for shell center
         */
        return m_R;
    }

    arma::vec get_alphas() const {
        /* Getter function for alpha parameters
         */
        return m_alphas;
    }

    arma::vec get_contractions() const {
        /* Getter function for contraction coefficients
         */
        return m_contractions;
    }

    arma::vec get_normalizations() const {
        /* Getter function for contraction coefficients
         */
        return m_normalizations;
    }

    arma::vec get_momentum() const {
        /* Getter function for momentum
         */
        return m_momentum;
    }



 private:
    string m_name;
    Atom* m_atom = nullptr;
    arma::vec m_R;
    arma::vec m_momentum;
    arma::vec m_alphas;
    arma::vec m_contractions;
    arma::vec m_normalizations;
};

#endif //BASISFUNCTION_H