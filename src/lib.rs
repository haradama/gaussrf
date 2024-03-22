use nalgebra::{DMatrix, DVector};
use rand_distr::{Distribution, Normal, Uniform};

struct GPR {
    dim: usize,
    std_kernel: f64,
    std_error: f64,
    w: Option<DMatrix<f64>>,
    b: Option<DVector<f64>>,
    m: Option<DMatrix<f64>>,
    a: Option<DVector<f64>>,
}

impl GPR {
    pub fn new(dim_kernel: usize, std_kernel: f64, std_error: f64) -> GPR {
        GPR {
            dim: dim_kernel,
            std_kernel,
            std_error,
            w: None,
            b: None,
            m: None,
            a: None,
        }
    }

    fn initialize_rff(&mut self, dim_in: usize) {
        let normal_dist = Normal::new(0.0, self.std_kernel).unwrap();
        let uniform_dist = Uniform::new(0.0, 2.0 * std::f64::consts::PI);
        let mut rng = rand::thread_rng();

        if self.w.is_none() {
            let w_data: Vec<f64> = (0..dim_in * self.dim)
                .map(|_| normal_dist.sample(&mut rng))
                .collect();
            self.w = Some(DMatrix::from_vec(dim_in, self.dim, w_data));
        }

        if self.b.is_none() {
            let b_data: Vec<f64> = (0..self.dim)
                .map(|_| uniform_dist.sample(&mut rng))
                .collect();
            self.b = Some(DVector::from_vec(b_data));
        }
    }

    fn apply_rff(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let w = self.w.as_ref().unwrap();
        let b = self.b.as_ref().unwrap();
        let mut phi = x * w;
        phi.column_iter_mut().enumerate().for_each(|(j, mut col)| {
            col.iter_mut().for_each(|val| *val = (*val + b[j]).cos());
        });
        phi
    }

    fn fit(&mut self, x: &DMatrix<f64>, y: &DVector<f64>) {
        self.initialize_rff(x.ncols());
        let phi = self.apply_rff(x);
        let phi_t = phi.transpose();
        let phi_t_phi = &phi_t * &phi;
        let p = phi_t_phi + DMatrix::identity(self.dim, self.dim) * self.std_error.powi(2);

        let m_inv = p.try_inverse().expect("Matrix inversion failed");
        self.m = Some(m_inv.clone());
        self.a = Some(m_inv * &phi_t * y);
    }
    
    fn predict(&self, x: &DMatrix<f64>) -> (DVector<f64>, DVector<f64>) {
        let phi = self.apply_rff(x).transpose();
        let mean = self.a.as_ref().unwrap().transpose() * &phi;
        let var = phi.transpose() * self.m.as_ref().unwrap() * &phi;
        let std = var.diagonal().map(|v| v.sqrt());
        (mean.transpose(), std)
    }

    fn score(&self, x: &DMatrix<f64>, y: &DVector<f64>) -> f64 {
        let (y_pred, _) = self.predict(x);
        let ss_res = (y - &y_pred).norm_squared();
        let y_mean = y.mean();
        let ss_tot = y.iter().map(|&val| (val - y_mean).powi(2)).sum::<f64>();
        1.0 - ss_res / ss_tot
    }
    
    fn update(&mut self, x_new: &DMatrix<f64>, y_new: &DVector<f64>) {
        let phi_new = self.apply_rff(x_new).transpose();
        let m_inv = self.m.as_ref().unwrap().clone().try_inverse().unwrap();
        let m_updated = (m_inv + &phi_new * phi_new.transpose()).try_inverse().unwrap();
        self.m = Some(m_updated.clone());
    
        let a_updated = self.a.as_ref().unwrap().clone() 
                      + m_updated * &phi_new * (y_new - &(phi_new.transpose() * self.a.as_ref().unwrap()));
        self.a = Some(a_updated);
    }    
}
