#include <iomanip>

#include "GeometryOptimizer.h"

void GeometryOptimizer::optimize() {
    /* Optimize based on given method
     */
    if (m_method =="sd") {
        run_steepest_descent();
    }
    else if (m_method == "bfgs") {
        run_bfgs();
    }
}

void put_coords_in_atoms(const arma::mat& pos, Molecule* mol) {
    /* Put the given coordinates into the atoms of the given molcule
     */
    std::vector<Atom>& atoms = mol->get_atoms();
    int num_atoms = atoms.size();
    assert(pos.n_cols == num_atoms && pos.n_rows == 3);

    for (int i = 0; i < atoms.size(); i++) {
        atoms[i].set_R(pos.col(i));
    }
}

double GeometryOptimizer::linesearch_wolfe(arma::vec& direction) {
    /* IN PROGRESS: Perform an inexact linesearch to find a stepsize that satisfies the Wolfe conditions
     */
    double c1 = 1E-4;
    double c2 = 0.9;
    double alpha = 0.0;
    double t = 1.;
    double beta = -1.;

    double tol = 1E-4;

    int iter = 0;

    while (alpha != t && t != beta) {
        Molecule temp = *m_mol;
        arma::vec deriv = vectorise(m_mol->get_analytic_gradient_AU());
        //assert(as_scalar(direction.t() * vectorise(temp.get_analytic_gradient_AU())) < -tol);
        double energy = m_mol->get_total_energy();
        arma::vec pos = arma::vectorise(m_pos) + t * direction;
        //pos.print("position");
        put_coords_in_atoms(arma::reshape(pos, m_pos.n_rows, m_pos.n_cols), &temp);
        temp.run();
        //std::cout << temp.get_total_energy() << " " <<  energy << " " << c1 * t * as_scalar(direction.t() * deriv) << std::endl; 
        //std::cout << as_scalar(direction.t() * vectorise(temp.get_analytic_gradient_AU())) << c2 * as_scalar(direction.t() * deriv) << std::endl;
        if (temp.get_total_energy() - (energy + c1 * t * as_scalar(direction.t() * deriv)) > tol) {
            //std::cout << temp.get_total_energy() << " " <<  energy + c1 * t * as_scalar(direction.t() * deriv) << std::endl;
            //std::cout << "condition i" << std::endl;
            beta = t;
            t = 0.5 * (alpha + beta);
        }
        else if (as_scalar(direction.t() * vectorise(temp.get_analytic_gradient_AU())) - c2 * as_scalar(direction.t() * deriv) < -tol) {
            //std::cout << as_scalar(direction.t() * vectorise(temp.get_analytic_gradient_AU())) << " " << c2 * as_scalar(direction.t() * deriv) << std::endl;
            //std::cout << "condition ii" << std::endl;
            alpha = t;
            if (beta==-1.) {
                t = 2. * alpha;
            }
            else {
                t = 0.5 * (alpha + beta);
            }
        }
        else {
            return t;
        }
        iter++;
        std::cout << "iteration " << iter << " t " << t << " alpha " << alpha <<  " beta " << beta << std::endl;

    }

    return t;

}

void GeometryOptimizer::run_bfgs() {
    /* Perform BFGS to optimize molecule geometry
     */
    //Initialize Hessian guess to identity
    arma::mat hess = arma::eye(m_pos.n_cols * m_pos.n_rows, m_pos.n_cols * m_pos.n_rows);
    arma::mat hess_inv = inv(hess);

    arma::vec pos = vectorise(m_pos);

    arma::vec deriv = vectorise(m_mol->get_analytic_gradient_AU());

    int count = 0;

    double initial_E = m_mol->get_total_energy();

    std::cout << "Starting BFGS..." << std::endl;
    // While the norm of the gradient is nonzero, or maximum iterations are reached
    while (arma::norm(deriv, "fro") > m_tol && count < m_max_iter) {
        // deriv.print("Deriv");
        // m_pos.print("Position");
        // hess.print("Hessian");
        // hess_inv.print("Hessian inverse");

        //Compute search direction
        arma::vec p_k = -hess_inv * deriv;

        //m_stepsize = linesearch_wolfe(p_k);

        //Compute step
        arma::vec s_k = m_stepsize * p_k;

        //Move atoms according to computed step
        pos += s_k;

        m_pos = arma::reshape(pos, m_pos.n_rows, m_pos.n_cols);

        m_pos.print("Current geometry");

        //Put new coordinates into the molecule and re-run CNDO calculations
        put_coords_in_atoms(m_pos, m_mol);
        m_mol->run();

        //Retrieve new gradient and compute difference between new and old
        arma::vec new_deriv = vectorise(m_mol->get_analytic_gradient_AU());

        arma::vec y_k = new_deriv - deriv;

        deriv = new_deriv;

        //Update Hessian according to BFGS calculation
        hess  += (y_k * y_k.t()) / as_scalar(y_k.t() * s_k) - (hess * s_k * s_k.t() * hess.t()) / as_scalar(s_k.t() * hess * s_k);

        //Compute Hessian inverse difference
        //arma::mat hess_inv_diff = ((as_scalar(s_k.t() * y_k) + as_scalar(y_k.t() * hess_inv * y_k)) * (s_k * s_k.t())) / std::pow(as_scalar(s_k.t() * y_k), 2) - (hess_inv * y_k * s_k.t() + s_k * y_k.t() * hess_inv) / as_scalar(s_k.t() * y_k);

        //hess_inv_diff.print("Hessian inverse difference");

        //Update Hessian inverse
        hess_inv += ((as_scalar(s_k.t() * y_k) + as_scalar(y_k.t() * hess_inv * y_k)) * (s_k * s_k.t())) / std::pow(as_scalar(s_k.t() * y_k), 2) - (hess_inv * y_k * s_k.t() + s_k * y_k.t() * hess_inv) / as_scalar(s_k.t() * y_k);

        count += 1;

        std::cout << "iteration: " << count << " Energy: " << std::setprecision(6) << m_mol->get_total_energy() << std::endl;
    }

    std::cout << "BFGS has concluded after " << count << " iterations." << std::endl;
    std::cout << "Initial energy: " << initial_E << " eV." << std::endl;
    std::cout << "Final energy: " << m_mol->get_total_energy() << " eV." << std::endl;


}

void GeometryOptimizer::run_steepest_descent() {
    /* Perform steepest descent to optimize atom geometry
     */
    int num_atoms = m_pos.n_cols;

    // Calculate derivative by central difference on initial atoms
    arma::mat deriv = m_mol->get_analytic_gradient_AU();
    // Retrieve the starting point coordinates from the initial atoms
    arma::mat& starting_point = m_pos;
    // Initialize matrix to store new points, size 3 x num_atoms
    arma::mat new_point(3, num_atoms);

    // Initialize count
    int count = 0;
    // Calculate energy at initial position
    double E_starting = m_mol->get_total_energy();
    // Initialize double to store new energies
    double E_new = 0.0;

    double initial_E = E_starting;

    std::cout << "Starting steepest descent..." << std::endl;
    // While the norm of the gradient is nonzero, or maximum iterations are reached
    while (arma::norm(deriv, "fro") > m_tol && count < m_max_iter) {
        //deriv.print("Deriv");
        // Calculate new atom positions
        new_point = starting_point - m_stepsize * deriv / arma::norm(deriv, 2);
        //new_point.print("New point: ");
        // Put the new coordinates into optimized_atoms
        put_coords_in_atoms(new_point, m_mol);
        m_mol->run();
        // Calculate the energy at the new coordinates
        E_new = m_mol->get_total_energy();
        //std::cout << E_LJ_new << std::endl;

        // If the new energy is less than the old, this is a good step
        if (E_new < E_starting) {
            // Accept the new coordinates
            starting_point = new_point;
            E_starting = E_new;
            // Calculate the gradient at the new point
            deriv = m_mol->get_analytic_gradient_AU();
            // Increase stepsize
            m_stepsize *= 1.2;
            std::cout << "Gradient norm: " << arma::norm(deriv, "fro") << std::endl;
        }
        // Bad step
        else {
            // Reject the new coordinates, restore the old
            put_coords_in_atoms(starting_point, m_mol);
            // decrease the step size
            m_stepsize /= 2;
        }

        count += 1;

        starting_point.print("Current geometry");

        std::cout << "iteration: " << count << " Energy: " << std::setprecision(6) << E_starting << std::endl;
    }

    std::cout << "Steepest descent has concluded after " << count << " iterations." << std::endl;
    std::cout << "Initial energy: " << initial_E << " eV." << std::endl;
    std::cout << "Final energy: " << m_mol->get_total_energy() << " eV." << std::endl;
}