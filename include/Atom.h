#ifndef ATOM_H
#define ATOM_H

#include <cmath>
#include <vector>
#include <stdexcept>
#include <armadillo>
#include <string>

#include "BasisFunction.h"

using namespace std;

class Atom {
 public:
    //Constructor
    Atom(const arma::vec& R, const int& Z, const int& index) {
        //Center should be of 3 dimensions
        if (R.size() != 3) {
            throw std::invalid_argument("R must be a vector of size 3 containing the x, y, and z coordinates of the shell center.");
        }

        //Can only do H, C, N, O, and F
        if (Z != 1 && (Z < 6 || Z > 9)) {
            throw std::invalid_argument("Currently, the program only supports H, C, N, O, and F atoms.");
        }

        //Can only do H, C, N, O, and F
        if (index < 0) {
            throw std::invalid_argument("Atom index must be greater than 0.");
        }

        m_R = R;
        m_Z = Z;
        m_index = index;

        //Initialize vector to store functions based on the value of Z
        vector<BasisFunction> basis_functions;

        switch(Z) {
            case 1: {
                //H 1s shell function
                arma::vec alphas({3.42525091, 0.62391373, 0.16885540});
                arma::vec contractions({0.15432897, 0.53532814, 0.44463454});
                basis_functions.push_back(BasisFunction("H1s", m_R, arma::vec({0., 0., 0.}), alphas, contractions));
                m_Z_val = 1;
                break;
            }
            case 6:
                {
                //C 2s and 2p shell functions
                arma::vec alphas({2.94124940, 0.68348310, 0.22228990});
                arma::vec s_contractions({-0.09996723, 0.39951283, 0.70011547});
                arma::vec p_contractions({0.15591627, 0.60768372, 0.39195739});
                basis_functions.push_back(BasisFunction("C2s", m_R, arma::vec({0., 0., 0.}), alphas, s_contractions));
                basis_functions.push_back(BasisFunction("C2p", m_R, arma::vec({1., 0., 0.}), alphas, p_contractions));
                basis_functions.push_back(BasisFunction("C2p", m_R, arma::vec({0., 1., 0.}), alphas, p_contractions));
                basis_functions.push_back(BasisFunction("C2p", m_R, arma::vec({0., 0., 1.}), alphas, p_contractions));
                m_Z_val = 4;
                break;
            }
            case 7:
                {
                //N 2s and 2p shell functions
                arma::vec alphas({0.3780455879E+01, 0.8784966449E+00, 0.2857143744E+00});
                arma::vec s_contractions({-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00});
                arma::vec p_contractions({0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00});
                basis_functions.push_back(BasisFunction("N2s", m_R, arma::vec({0., 0., 0.}), alphas, s_contractions));
                basis_functions.push_back(BasisFunction("N2p", m_R, arma::vec({1., 0., 0.}), alphas, p_contractions));
                basis_functions.push_back(BasisFunction("N2p", m_R, arma::vec({0., 1., 0.}), alphas, p_contractions));
                basis_functions.push_back(BasisFunction("N2p", m_R, arma::vec({0., 0., 1.}), alphas, p_contractions));
                m_Z_val = 5;
                break;
            }
            case 8:
                {
                //O 2s and 2p shell functions
                arma::vec alphas({5.033151319, 1.169596125, 0.3803889600});
                arma::vec s_contractions({-0.09996722919,0.3995128261, 0.7001154689});
                arma::vec p_contractions({0.1559162750, 0.6076837186, 0.3919573931});
                basis_functions.push_back(BasisFunction("O2s", m_R, arma::vec({0., 0., 0.}), alphas, s_contractions));
                basis_functions.push_back(BasisFunction("O2p", m_R, arma::vec({1., 0., 0.}), alphas, p_contractions));
                basis_functions.push_back(BasisFunction("O2p", m_R, arma::vec({0., 1., 0.}), alphas, p_contractions));
                basis_functions.push_back(BasisFunction("O2p", m_R, arma::vec({0., 0., 1.}), alphas, p_contractions));
                m_Z_val = 6;
                break;
            }
            case 9:
                {
                //F 2s and 2p shell functions
                arma::vec alphas({0.6464803249E+01, 0.1502281245E+01, 0.4885884864E+00});
                arma::vec s_contractions({-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00});
                arma::vec p_contractions({0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00});
                basis_functions.push_back(BasisFunction("F2s", m_R, arma::vec({0., 0., 0.}), alphas, s_contractions));
                basis_functions.push_back(BasisFunction("F2p", m_R, arma::vec({1., 0., 0.}), alphas, p_contractions));
                basis_functions.push_back(BasisFunction("F2p", m_R, arma::vec({0., 1., 0.}), alphas, p_contractions));
                basis_functions.push_back(BasisFunction("F2p", m_R, arma::vec({0., 0., 1.}), alphas, p_contractions));
                m_Z_val = 7;
                break;
            }
            default:
                break;
        }

        m_basis_functions = basis_functions;
    }

    arma::vec get_R() const {
        /* Getter function for atom position
         */
        return m_R;
    }

    void set_R(const arma::vec& R) {
        if (R.size() != 3) {
            throw std::invalid_argument("R must be a vector of size 3 containing the x, y, and z coordinates of the shell center.");
        }
        m_R = R;
        for (int i = 0; i < m_basis_functions.size(); i++) {
            m_basis_functions[i].set_R(R);
        }
    }

    int get_Z() const {
        /* Getter function for atomic number
         */
        return m_Z;
    }

    int get_Z_val() const {
        /* Getter function for atomic number
         */
        return m_Z_val;
    }

    int get_index() const {
        /* Getter function for atom index
         */
        return m_index;
    }

    vector<BasisFunction>& get_basis_functions() {
        /* Getter function for basis functions
         */
        return m_basis_functions;
    }

 private:
    int m_index;
    arma::vec m_R;
    int m_Z;
    int m_Z_val;
    std::vector<BasisFunction> m_basis_functions;
};

#endif //ATOM_H