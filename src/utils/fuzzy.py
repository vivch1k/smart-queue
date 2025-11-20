import skfuzzy as fuzz
import numpy as np

class CashWorkload():
    def __init__(self, A, B, a_term_l, a_term_m, a_term_h, b_term_l, b_term_m, b_term_h, rules):
        self.A = A
        self.B = B
        self.a_term_l = a_term_l
        self.a_term_m = a_term_m
        self.a_term_h = a_term_h
        self.b_term_l = b_term_l
        self.b_term_m = b_term_m
        self.b_term_h = b_term_h

        self.u_a_l = fuzz.trimf(self.A, self.a_term_l)
        self.u_a_m = fuzz.trimf(self.A, self.a_term_m)
        self.u_a_h = fuzz.trimf(self.A, self.a_term_h)

        self.u_b_l = fuzz.trimf(self.B, self.b_term_l)
        self.u_b_m = fuzz.trimf(self.B, self.b_term_m)
        self.u_b_h = fuzz.trimf(self.B, self.b_term_h)

        self.rules = rules

    def sugeno(self, a_val, b_val):
        uA = [fuzz.interp_membership(self.A, self.u_a_l, a_val),
              fuzz.interp_membership(self.A, self.u_a_m, a_val),
              fuzz.interp_membership(self.A, self.u_a_h, a_val)]
    
        uB = [fuzz.interp_membership(self.B, self.u_b_l, b_val),
              fuzz.interp_membership(self.B, self.u_b_m, b_val),
              fuzz.interp_membership(self.B, self.u_b_h, b_val)]
        
        w = np.zeros((3, 3))
        z = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                w[i, j] = uA[i] * uB[j]
                z[i, j] = self.rules[i, j]

        return np.sum(w * z) / np.sum(w)