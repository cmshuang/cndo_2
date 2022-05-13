#include "SCFFunctions.h"
#include <armadillo>
#include <iomanip>

void BasisFunction::calculate_normalizations() {
    /* Compute normalization constants for the basis function
     */
    arma::vec normalizations(m_alphas.size());
    for (int k = 0; k < m_alphas.size(); k++) {
        normalizations(k) = sqrt(1. / compute_S_ab(m_R, m_R, m_momentum, m_momentum, m_alphas(k), m_alphas(k))); // N^2 * S_aa = 1 - > N = sqrt(1 / S_aa)
    }
    m_normalizations = normalizations;
}

void Molecule::write_output(const string& filename) {
    std::ofstream ofile;
    ofile.open(filename);
    if (ofile.is_open()) {
        // Set output precision to 16 decimal places (floating point tol)
        ofile << m_atoms.size() << std::endl;
        ofile << std::scientific << std::setprecision(16);
        // Write each matrix element to file, space delimited
        for (int i = 0; i < m_atoms.size(); i++) {
            ofile << m_atoms[i].get_Z();
            for (int d = 0; d < 3; d++) {
                ofile << " " << m_atoms[i].get_R()[d];
            }
            ofile << std::endl;
        }
    }
}

void Molecule::calculate_overlap_matrix() {
    /* Compute overlap matrix of the molecule
     */
    arma::mat overlap_matrix(m_N, m_N);

    // N x N matrix
    for (int i = 0; i < m_N; i++) {
        // Matrix is symmetric
        for (int j = i; j < m_N; j++) {
            // See hw3 pdf eq 2.5 for formula
            double S_ij = 0.;
            BasisFunction* omega_i = m_all_basis_functions[i];
            BasisFunction* omega_j = m_all_basis_functions[j];
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    S_ij += omega_i->get_contractions()[k] * omega_j->get_contractions()[l] * omega_i->get_normalizations()[k] * omega_j->get_normalizations()[l] * compute_S_ab(omega_i->get_R(), omega_j->get_R(), omega_i->get_momentum(), omega_j->get_momentum(), omega_i->get_alphas()[k], omega_j->get_alphas()[l]);
                }
            }
            overlap_matrix(i, j) = S_ij;
            overlap_matrix(j, i) = S_ij;
        }
    }
    m_S = overlap_matrix;

    //m_S.print("Overlap matrix");
}


void Molecule::calculate_gamma() {
    /* Assemble the gamma matrix for the molecule
     */
    int num_atoms = m_atoms.size();
    for (int i = 0; i < num_atoms; i++) {
        for (int j = 0; j < num_atoms; j++) {
            m_gamma(i, j) = calculate_gamma_AB(m_atoms[i], m_atoms[j]);
        }
    }
    //m_gamma.print("Gamma matrix");
    return;
}

void Molecule::calculate_p_tot_atom() {
    /* Assemble the vector of atomwwise total density
     */
    m_p_tot_atom = arma::vec(m_atoms.size(), arma::fill::zeros);
    // See eq 1.3, 1.6 in hw 4 pdf
    for (int i = 0; i < m_N; i++) {
        Atom* A = m_all_basis_functions[i]->get_atom();
        assert(A != nullptr);
        m_p_tot_atom(A->get_index()) += m_p_alpha(i, i) + m_p_beta(i, i);
    }
}

void Molecule::perform_SCF() {
    /* Perform the SCF algorithm
     */
    double tol=1E-7;
    // Initial guess is p_alpha = p_beta = 0
    arma::mat p_alpha_old = m_p_alpha;
    arma::mat p_beta_old = m_p_beta;
    std::cout << "Starting SCF iterations..." << std::endl;
    int iterations = 0;
    do {
        // Copy the density matrix to old
        p_alpha_old = m_p_alpha;
        p_beta_old = m_p_beta;
        // Calculate the fock matrices given the density matrices
        m_f_alpha = calculate_fock_matrix(m_all_basis_functions, m_S, m_p_alpha, m_atoms, m_p_tot_atom, m_gamma);
        m_f_beta = calculate_fock_matrix(m_all_basis_functions, m_S, m_p_beta, m_atoms, m_p_tot_atom, m_gamma);
        // Solve eigenvalue problem to fill MO coefficients and epsilons for alpha and beta
        arma::eig_sym(m_epsilon_alpha, m_C_alpha, m_f_alpha);
        arma::eig_sym(m_epsilon_beta, m_C_beta, m_f_beta);
        //Calculate the new density matrices from the new MO coefficients
        m_p_alpha = calculate_density_matrix(m_C_alpha, m_p);
        m_p_beta = calculate_density_matrix(m_C_beta, m_q);
        calculate_p_tot_atom();
        iterations++;
    } while((abs((m_p_alpha - p_alpha_old).max()) > tol || abs((m_p_beta - p_beta_old).max()) > tol) && iterations < 1E+2); // Iterate until convergence or maximum number of steps reached
    std::cout << "After " << iterations << " iterations, SCF is converged to " << tol << std::endl;
    // m_p_alpha.print("P_alpha");
    // m_p_beta.print("P_beta");
    // m_f_alpha.print("F_alpha");
    // m_f_beta.print("F_beta");
    // m_epsilon_alpha.print("E_alpha");
    // m_epsilon_beta.print("E_beta");
    // m_C_alpha.print("C_alpha");
    // m_C_beta.print("C_beta");
}

void DIIS(arma::mat &e, arma::vec &c) {
    /* Make the DIIS calculations
     */
    int rank = c.n_elem;
    assert(e.n_cols == rank - 1);
    arma::mat e_mat (rank, rank);
    arma::vec b =arma::zeros(rank);
    b(rank-1) = -1.;
    e_mat.col(rank -1).fill(-1.);
    e_mat.row(rank -1).fill(-1.);
    e_mat(rank -1, rank -1) = 0.;
    e_mat.submat( 0, 0, rank -2, rank -2) = e.t() *e;
    // e_mat.print("e_mat");
    c =arma::solve(e_mat, b);
}

void Molecule::perform_DIIS() {
    /* Perform the DIIS algorithm
     */
    int DIIS_circle = 4;
    int max_iter = 1E+3;
    double tol = 1E-7;
    m_f_alpha = calculate_fock_matrix(m_all_basis_functions, m_S, m_p_alpha, m_atoms, m_p_tot_atom, m_gamma);
    m_f_beta = calculate_fock_matrix(m_all_basis_functions, m_S, m_p_beta, m_atoms, m_p_tot_atom, m_gamma);
    arma::mat p_alpha_old, p_beta_old;
    arma::mat f_a_record(m_N * m_N, DIIS_circle), f_b_record(m_N * m_N, DIIS_circle);
    arma::mat ea(m_N * m_N, DIIS_circle), eb(m_N * m_N, DIIS_circle);
    size_t k = 0;
    std::cout << "Starting DIIS iterations..." << std::endl;
    for (; k < max_iter; k++) {
        p_alpha_old = m_p_alpha;
        p_beta_old = m_p_beta;
        // Solve eigenvalue problem to fill MO coefficients and epsilons for alpha and beta
        arma::eig_sym(m_epsilon_alpha, m_C_alpha, m_f_alpha);
        arma::eig_sym(m_epsilon_beta, m_C_beta, m_f_beta);
        //Calculate the new density matrices from the new MO coefficients
        m_p_alpha = calculate_density_matrix(m_C_alpha, m_p);
        m_p_beta = calculate_density_matrix(m_C_beta, m_q);
        calculate_p_tot_atom();
        if (abs((m_p_alpha - p_alpha_old).max()) < tol && abs((m_p_beta - p_beta_old).max()) < tol)
            break;

        m_f_alpha = calculate_fock_matrix(m_all_basis_functions, m_S, m_p_alpha, m_atoms, m_p_tot_atom, m_gamma);
        m_f_beta = calculate_fock_matrix(m_all_basis_functions, m_S, m_p_beta, m_atoms, m_p_tot_atom, m_gamma);

        int k_DIIS = k % DIIS_circle;
        arma::mat fa_r(f_a_record.colptr(k_DIIS), m_N, m_N, false, true);
        arma::mat fb_r(f_b_record.colptr(k_DIIS), m_N, m_N, false, true);
        fa_r = m_f_alpha; fb_r = m_f_beta;
        arma::mat ea_r(ea.colptr(k_DIIS), m_N, m_N, false, true);
        arma::mat eb_r(eb.colptr(k_DIIS), m_N, m_N, false, true);
        ea_r = m_f_alpha * m_p_alpha - m_p_alpha * m_f_alpha;
        ea.col(k_DIIS) = ea_r.as_col();
        // ea_r.print("ea");
        eb_r = m_f_beta * m_p_beta - m_p_beta * m_f_beta;
        if (k_DIIS == DIIS_circle - 1 && k / DIIS_circle > 0) {
            arma::vec ca(DIIS_circle + 1), cb(DIIS_circle + 1);
            DIIS(ea, ca);
            DIIS(eb, cb);
            // ca.print("ca");
            arma::vec f_a_vec(m_f_alpha.memptr(), m_N * m_N, false, true);
            arma::vec f_b_vec(m_f_beta.memptr(), m_N * m_N, false, true);
            f_a_vec = f_a_record.col(0) *ca(0);
            f_b_vec = f_b_record.col(0) *cb(0);
            for(size_t j = 1; j< DIIS_circle; j++) {
                f_a_vec += f_a_record.col(j) *ca(j);
                f_b_vec += f_b_record.col(j) *ca(j);
            }
        }
        
    }
    if (k == max_iter) {
        cout << "Error: the job could not be finished in " << max_iter << "iterations.\n";
        return;
    }
    std::cout << "After " << k << " iterations, SCF is converged to " << tol << std::endl;
}

void Molecule::calculate_total_energy() {
    /* Assemble the core Hamiltonian matrix and calculate total energy
     */
    arma::mat H(m_N, m_N, arma::fill::zeros);

    // Iterate through AO basis to compute core Hamiltonian matrix
    for (int i = 0; i < m_N; i++) {
        for (int j = 0; j < m_N; j++) {
            BasisFunction* omega_i = m_all_basis_functions[i];
            BasisFunction* omega_j = m_all_basis_functions[j];
            Atom* A = omega_i->get_atom();
            Atom* B = omega_j->get_atom();
            assert(A != nullptr && B != nullptr);
            // Diagonal element, see eq 2.6 in hw 4 pdf
            if (i == j) {
                H(i, i) = -CNDO_param_map[omega_i->get_name()] - ((double)A->get_Z_val() - 0.5) * m_gamma(A->get_index(), A->get_index());
                for (int k = 0; k < m_atoms.size(); k++) {
                    if (k != A->get_index()) {
                        H(i, j) -= m_atoms[k].get_Z_val() * m_gamma(A->get_index(), k);
                    }
                }
            }
            // Off-diagonal element, see eq 2.6 in hw 4 pdf
            else {
                H(i, j) = -0.5 * (CNDO_beta_param_map[A->get_Z()] + CNDO_beta_param_map[B->get_Z()]) * m_S(i, j);
            }
        }
    }
    m_H = H;
    // H.print("H_core");

    // See eq 2.5 in hw 4 pdf for full energy calculation
    double total_energy = 0.;

    // Calculate the electron energy
    for (int i = 0; i < m_N; i++) {
        for (int j = 0; j < m_N; j++) {
            total_energy += m_p_alpha(i, j) * (m_H(i, j) + m_f_alpha(i, j)) + m_p_beta(i, j) * (m_H(i, j) + m_f_beta(i, j));
        }
    }
    total_energy *= 0.5;

    // Calculate nuclear repulsion energy
    double nuclear_repulsion_E = 0.;
    for (int i = 0; i < m_atoms.size(); i++) {
        for (int j = 0; j < i; j++) {
            double R = std::sqrt(arma::sum(arma::pow((m_atoms[i].get_R() - m_atoms[j].get_R()), 2)));
            nuclear_repulsion_E += m_atoms[i].get_Z_val() * m_atoms[j].get_Z_val() / R;
        }
    }
    nuclear_repulsion_E *= AU_TO_EV_CONVERSION;
    m_total_energy = total_energy + nuclear_repulsion_E;

    std::cout << std::setprecision(6) << "Nuclear repulsion Energy: " << nuclear_repulsion_E << " eV" << std::endl;
    std::cout << std::setprecision(6) << "Total Energy: " << m_total_energy << " eV" << std::endl;
}

void Molecule::calculate_analytic_gradient_E_finite_difference() {
    // Initialize matrix to store forces, of dimensions 3 x num_atoms
    arma::mat forces(3, m_atoms.size());
    double h = 0.00001;

    // For each atom
    for (int k = 0; k < m_atoms.size(); k++) {
        // For each dimension
        for (int d = 0; d < 3; d++) {
            // Copy original atoms vector into new vectors, one to store the forward difference and one to store the backward
            arma::vec original_R = m_atoms[k].get_R();
            arma::vec forward_R = original_R;
            forward_R[d] += h;
            m_atoms[k].set_R(forward_R);
            this->run();
            double E_forwards = m_total_energy;
            // Move atom k in the forward vector in the positive direction by h
            // Move atom k in the backward vector in the negative direction by h
            arma::vec backward_R = original_R;
            backward_R[d] -= h;
            m_atoms[k].set_R(backward_R);
            this->run();
            double E_backwards = m_total_energy;
            // F_k = -(2*h)^-1 * [E(R_i + h) - E(R_i - h)]
            forces(d, k) = -(E_forwards - E_backwards) / (2 * h);
            std::cout << forces(d, k) << std::endl;
            m_atoms[k].set_R(original_R);
            this->run();
        }
    }

    m_analytic_gradient_E = forces;
    m_analytic_gradient_E.print("Analytic gradient");
}

void Molecule::calculate_x() {
    /* Assemble the x matrix for analytical gradient calculation
     */
    for (int i = 0; i < m_N; i++) {
        // Matrix is symmetric
        for (int j = i; j < m_N; j++) {
            // See eq 1.17 in hw 5 pdf
            BasisFunction* omega_i = m_all_basis_functions[i];
            BasisFunction* omega_j = m_all_basis_functions[j];
            Atom* A = omega_i->get_atom();
            Atom* B = omega_j->get_atom();
            assert(A != nullptr && B != nullptr);
            m_x(i, j) = -(CNDO_beta_param_map[A->get_Z()] + CNDO_beta_param_map[B->get_Z()]) * (m_p_alpha(i, j) + m_p_beta(i, j));
            m_x(j, i) = m_x(i, j);
        }
    }
}

void Molecule::calculate_y() {
    /* Assemble the y matrix for analytical gradient calculation
     */
    for (int i = 0; i < m_atoms.size(); i++) {
        // Matrix is symmetric
        for (int j = i; j < m_atoms.size(); j++) {
            // See eq 1.18 in hw 5 pdf
            m_y(i, j) = m_p_tot_atom[i] * m_p_tot_atom[j] - m_atoms[j].get_Z_val() * m_p_tot_atom[i] - m_atoms[i].get_Z_val() * m_p_tot_atom[j];
            for (int mu = 0; mu < m_N; mu++) {
                for (int nu = 0; nu < m_N; nu++) {
                    BasisFunction* omega_mu = m_all_basis_functions[mu];
                    BasisFunction* omega_nu = m_all_basis_functions[nu];
                    Atom* A = omega_mu->get_atom();
                    Atom* B = omega_nu->get_atom();
                    assert(A != nullptr && B != nullptr);
                    int atom_mu = A->get_index();
                    int atom_nu = B->get_index();
                    if (atom_mu == i && atom_nu == j) {
                        m_y(i, j) -= m_p_alpha(mu, nu) * m_p_alpha(mu, nu) + m_p_beta(mu, nu) * m_p_beta(mu, nu);
                    }
                }
            }
            m_y(j, i) = m_y(i, j);
        }
    }
}

void Molecule::calculate_analytic_gradient_E() {
    /* Compute the analytic gradient of the SCF energy. See eq 1.16 in hw 5 pdf
     */
    calculate_x();
    calculate_y();

    // m_x.print("x");
    // m_y.print("y");

    for (int k = 0; k < m_atoms.size(); k++) {
        for (int d = 0; d < 3; d++) {
            arma::mat overlap_gradient = calculate_overlap_gradient_matrix_Xa(m_all_basis_functions, k, d);
            // std::string overlap_name = "Overlap gradient for atom " + std::to_string(k) + " in dimension " + std::to_string(d);
            // overlap_gradient.print(overlap_name);
            arma::mat gamma_gradient = calculate_gamma_gradient_Xa(m_atoms, k, d);
            // std::string gamma_name = "Gamma gradient for atom " + std::to_string(k) + " in dimension " + std::to_string(d);
            // gamma_gradient.print(gamma_name);

            //Terms involving overlap integral derivatives and x
            for (int i = 0; i < m_N; i++) {
                for (int j = 0; j < m_N; j++) {
                    if (i == j) {
                        continue;
                    }
                    BasisFunction* omega_i = m_all_basis_functions[i];
                    BasisFunction* omega_j = m_all_basis_functions[j];
                    Atom* A = omega_i->get_atom();
                    Atom* B = omega_j->get_atom();
                    assert(A != nullptr && B != nullptr);
                    int atom_i = A->get_index();
                    int atom_j = B->get_index();
                    if (atom_i == k && atom_j != k) {
                        m_analytic_gradient_E(d, k) += m_x(i, j) * overlap_gradient(i, j);
                    }
                }
            }

            for (int i = 0; i < m_atoms.size(); i++) {
                if (i != k) {
                    m_analytic_gradient_E(d, k) += m_y(k, i) * gamma_gradient(k, i); // Term involving gamma derivative and y
                    double R2 = arma::sum(arma::pow((m_atoms[k].get_R() - m_atoms[i].get_R()), 2));
                    // Nuclear repulsion derivative
                    m_analytic_gradient_E(d, k) +=  -m_atoms[k].get_Z_val() * m_atoms[i].get_Z_val() * std::pow(R2, -3./2.) * (m_atoms[k].get_R()[d] - m_atoms[i].get_R()[d]) * AU_TO_EV_CONVERSION;
                }
            }
        }
    }

    m_analytic_gradient_E.print("Analytic gradient");
}