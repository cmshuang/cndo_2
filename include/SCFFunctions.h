#ifndef HUCKEL_FUNCTIONS_H
#define HUCKEL_FUNCTIONS_H

#include <cmath>
#include <vector>
#include <stdexcept>
#include <armadillo>
#include <string>

//#include "BasisFunction.h"
#include "Molecule.h"

extern double AU_TO_EV_CONVERSION;

extern map<string, double> CNDO_param_map;
extern map<int, double> CNDO_beta_param_map;
extern double K;

Molecule read_molecule(const std::string& filename, const bool& toDIIS);
arma::vec compute_R_p(const double& alpha, const double& beta, const arma::vec& R_a, const arma::vec& R_b);
arma::vec compute_prefactor(const double& alpha, const double& beta, const arma::vec& R_a, const arma::vec& R_b);
double compute_S_ab_x(const double& alpha, const double& beta, const double& prefactor, const double& x_p, const double& x_a, const double& x_b, const double& l_a, const double& l_b);
double compute_S_ab(const arma::vec& R_a, const arma::vec& R_b, const arma::vec& momentum_a, const arma::vec& momentum_b, const double& alpha, const double& beta);
double calculate_gamma_AB(Atom A, Atom B);
arma::mat calculate_fock_matrix(const std::vector<BasisFunction*>& basis_functions, const arma::mat& S, const arma::mat& p, const std::vector<Atom>& atoms, const arma::vec& p_tot_atom, const arma::mat& gamma);
arma::mat calculate_density_matrix(const arma::mat& C, int num_lowest);
double compute_dS_ab_dXa(const arma::vec& R_a, const arma::vec& R_b, const arma::vec& momentum_a, const arma::vec& momentum_b, const double& alpha, const double& beta, const int& dim);
arma::mat calculate_overlap_gradient_matrix_Xa(const std::vector<BasisFunction*>& basis_functions, const int& atom_index, const int& dim);
double calculate_dgamma_AB_dXa(Atom A, Atom B, const int& dim);
arma::mat calculate_gamma_gradient_Xa(const std::vector<Atom>& atoms, const int& atom_index, const int& dim);

#endif //HUCKEL_FUNCTIONS_H