#ifndef MOLECULE_H
#define MOLECULE_H

#include <cmath>
#include <vector>
#include <map>
#include <stdexcept>
#include <armadillo>
#include <string>
#include <cassert>

#include "Atom.h"

using namespace std;

class Molecule {
 public:
    void write_output(const string& filename);
    void calculate_overlap_matrix();
    void calculate_gamma();
    void calculate_p_tot_atom();
    void perform_DIIS();
    void perform_SCF();
    void calculate_total_energy();
    void calculate_analytic_gradient_E();
    void calculate_analytic_gradient_E_finite_difference();
    arma::mat get_analytic_gradient_AU() const;
    void run() {
        // for (int i = 0; i < m_N; i++) {
        //     m_all_basis_functions[i]->print_info();
        // }
        arma::mat p_initialization(m_N, m_N, arma::fill::zeros);
        m_p_alpha = p_initialization;
        m_p_beta = p_initialization;

        m_p_tot_atom = arma::vec(m_atoms.size(), arma::fill::zeros);
        m_gamma = arma::mat(m_atoms.size(), m_atoms.size(), arma::fill::zeros);

        m_analytic_gradient_E = arma::mat(3, m_atoms.size(), arma::fill::zeros);
        m_x = arma::mat(m_N, m_N, arma::fill::zeros);
        m_y = arma::mat(m_atoms.size(), m_atoms.size(), arma::fill::zeros);
        calculate_overlap_matrix();
        calculate_gamma();
        if (m_DIIS)
            perform_DIIS();
        else
            perform_SCF();
        calculate_total_energy();
        calculate_analytic_gradient_E();
        //calculate_analytic_gradient_E_finite_difference();
    }
    //Constructor
    Molecule(vector<Atom>& atoms, const int& N, const int& num_electrons, const bool& DIIS): m_DIIS(DIIS) {
        //atoms cannot be empty
        if (atoms.empty()) {
            throw std::invalid_argument("Vector must contain at least one atom.");
        }

        //Can only do H and C
        if (N <= 0 || num_electrons <= 0) {
            throw std::invalid_argument("There must be a positive number of AOs and electrons.");
        }

        m_atoms = atoms;

        // Create vector of pointers to BasisFunctions
        vector<BasisFunction*> all_basis_functions;
        for (int i = 0; i < m_atoms.size(); i++) {
            assert(m_atoms[i].get_index() == i); // Make sure atoms are in order
            vector<BasisFunction>& basis_functions = m_atoms[i].get_basis_functions();
            for (int j = 0; j < basis_functions.size(); j++) {
                basis_functions[j].set_atom(&m_atoms[i]); //Set the atom pointer of the basis function
                all_basis_functions.push_back(&basis_functions[j]);
            }
        }

        assert(all_basis_functions.size() == N);

        m_all_basis_functions = all_basis_functions;

        m_N = N;
        m_q = num_electrons / 2;
        m_p = num_electrons - m_q;
        std::cout << "p = " << m_p << " q = " << m_q << std::endl;

        run();
        //calculate_analytic_gradient_E_finite_difference();
    }

    //Copy constructor
    Molecule(const Molecule &m1) {
        m_N = m1.m_N;
        m_p = m1.m_p;
        m_q = m1.m_q;
        m_DIIS = m1.m_DIIS;
        m_atoms = m1.m_atoms;

        // Create vector of pointers to BasisFunctions
        vector<BasisFunction*> all_basis_functions;
        for (int i = 0; i < m_atoms.size(); i++) {
            assert(m_atoms[i].get_index() == i); // Make sure atoms are in order
            vector<BasisFunction>& basis_functions = m_atoms[i].get_basis_functions();
            for (int j = 0; j < basis_functions.size(); j++) {
                basis_functions[j].set_atom(&m_atoms[i]); //Set the atom pointer of the basis function
                all_basis_functions.push_back(&basis_functions[j]);
            }
        }
        
        m_all_basis_functions = all_basis_functions;

        m_analytic_gradient_E = m1.m_analytic_gradient_E;
        m_x = m1.m_x;
        m_y = m1.m_y;
        m_gamma = m1.m_gamma;
        m_S = m1.m_S;
        m_H = m1.m_H;
        m_C_alpha = m1.m_C_alpha;
        m_C_beta = m1.m_C_beta;
        m_f_alpha = m1.m_f_alpha;
        m_f_beta = m1.m_f_beta;
        m_p_alpha = m1.m_p_alpha;
        m_p_beta = m1.m_p_beta;
        m_p_tot_atom = m1.m_p_tot_atom;
        m_epsilon_alpha = m1.m_epsilon_alpha;
        m_epsilon_beta = m1.m_epsilon_beta;
        m_total_energy = m1.m_total_energy;
    }

    std::vector<BasisFunction*> get_all_basis_functions() {
        return m_all_basis_functions;
    }

    int get_N() const {
        /* Getter function for number of basis functions
         */
        return m_N;
    }

    int get_p() const {
        /* Getter function for number of alpha electrons
         */
        return m_p;
    }

    int get_q() const {
        /* Getter function for number of beta electrons
         */
        return m_q;
    }

    double get_total_energy() const {
        /* Getter function for total energy
         */
        return m_total_energy;
    }

    vector<Atom> get_atoms() const {
        /* Getter function for basis functions
         */
        return m_atoms;
    }

    vector<Atom>& get_atoms() {
        /* Reference Getter function for basis functions
         */
        return m_atoms;
    }

    arma::mat get_analytic_gradient() const {
        /* Getter function for forces
         */
        return m_analytic_gradient_E;
    }

    arma::mat get_gamma() const {
        /* Getter function for gamma matrix
         */
        return m_gamma;
    }

    arma::mat get_S() const {
        /* Getter function for overlap matrix
         */
        return m_S;
    }

    arma::mat get_H() const {
        /* Getter function for core Hamiltonian
         */
        return m_H;
    }

    arma::mat get_C_alpha() const {
        /* Getter function for alpha molecular orbital coefficients
         */
        return m_C_alpha;
    }

    arma::mat get_C_beta() const {
        /* Getter function for beta molecular orbital coefficients
         */
        return m_C_beta;
    }

    arma::vec get_epsilon_alpha() const {
        /* Getter function for alpha energy eigenvalues
         */
        return m_epsilon_alpha;
    }

    arma::vec get_epsilon_beta() const {
        /* Getter function for beta energy eigenvalues
         */
        return m_epsilon_beta;
    }

 private:
    void calculate_x();
    void calculate_y();
    int m_N;
    int m_p;
    int m_q;
    bool m_DIIS;
    std::vector<Atom> m_atoms;
    std::vector<BasisFunction*> m_all_basis_functions;
    arma::mat m_analytic_gradient_E;
    arma::mat m_x;
    arma::mat m_y;
    arma::mat m_gamma;
    arma::mat m_S;
    arma::mat m_H;
    arma::mat m_C_alpha;
    arma::mat m_C_beta;
    arma::mat m_f_alpha;
    arma::mat m_f_beta;
    arma::mat m_p_alpha;
    arma::mat m_p_beta;
    arma::vec m_p_tot_atom;
    arma::vec m_epsilon_alpha;
    arma::vec m_epsilon_beta;
    double m_total_energy;
};

#endif //MOLECULE_H