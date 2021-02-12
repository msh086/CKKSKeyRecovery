#include "seal_utils.h"
#include <seal/util/uintarithsmallmod.h>
#include <seal/util/polyarithsmallmod.h>

#include <iostream>

void print_parameters(std::shared_ptr<seal::SEALContext> context) {
    auto &context_data = *context->key_context_data();
    std::cout << "Encryption parameters :" << std::endl;
    std::cout << "   poly_modulus_degree: " <<
        context_data.parms().poly_modulus_degree() << std::endl;
    // Print the size of the true (product) coefficient modulus.
    std::cout << "   coeff_modulus size: ";

    std::cout << context_data.total_coeff_modulus_bit_count() << " (";
    auto coeff_modulus = context_data.parms().coeff_modulus();
    std::size_t coeff_mod_count = coeff_modulus.size();
    for (std::size_t i = 0; i < coeff_mod_count - 1; i++)
    {
        std::cout << coeff_modulus[i].bit_count() << " + ";
    }

    std::cout << coeff_modulus.back().bit_count();
    std::cout << ") bits" << std::endl;
}

/**
 * NOTE on CRT and NTT
 *  let s be the product of all the coeff modulus(assume there are k modulus)
 *  decomposing the coeffs of a polynomial in R_s into a (k x N) matrix (CRT decompose)
 *  is a homomorphic mapping wrt addition & multiplication & inversion
 *  (the proof for inversion homomorphism can be easily derived from the multiplication homomorphism,
 *  while the proof for multiplication homomorphism can be done by simply writing coeffs out)
 *
 * the NTT operates in a similar way as DFT and canonical mapping:
 * evaluate the decomposed polynomials at w^0, w^1, ..., w^(N-1) where w is the N-primitive root modulo p_i
 * multiplication of polynomials becomes element-wise addition in the NTT domain
 * the matrix for transformation shares common properties as DFT & canonical mapping matrices:
 *  e.g. the product of it and its transpose conjugate is NI where I is the identity matrix
 * the homomorphism for addition, multiplication and inversion between first and second CRT layer
 * can be proved in a similar way as the canonical embedding, since w is a N-th primitive root modulo q
 * here the NTT modulus q is exactly the CRT modulus, since q mod N == 1 is guaranteed upon its generation,
 * see numth.cpp::get_primes line 294-317, where the modulus chain is generated
 * */
bool inv_dcrtpoly(util::ConstCoeffIter operand, std::size_t coeff_count, std::vector<Modulus> const& coeff_modulus,
                  util::CoeffIter result) {
   bool * has_inv = new bool[coeff_modulus.size()];
   std::fill_n(has_inv, coeff_modulus.size(), true);
#pragma omp parallel for
   for (size_t j = 0; j < coeff_modulus.size(); j++) {
      for (size_t i = 0; i < coeff_count && has_inv[j]; i++) {
         uint64_t inv = 0;
         if (util::try_invert_uint_mod(operand[i + (j * coeff_count)], coeff_modulus[j], inv)) {
            result[i + (j * coeff_count)] = inv;
         } else {
            has_inv[j] = false;
         }
      }
   }
   for (size_t j = 0; j < coeff_modulus.size(); j++) {
      if (!has_inv[j]) return false;
   }
   delete [] has_inv;
   return true;
}

/**
 * snippet for polynomial multiplication in NTT domain can be found at rlwe.cpp::encrypt_zero_asymmetric line 203-205
 * */
void mul_dcrtpoly(util::ConstCoeffIter a, util::ConstCoeffIter b, std::size_t coeff_count,
                  std::vector<Modulus> const& coeff_modulus, util::CoeffIter result) {
#pragma omp parallel for
   for (size_t j = 0; j < coeff_modulus.size(); j++) {
      util::dyadic_product_coeffmod(a + (j * coeff_count),
                                    b + (j * coeff_count),
                                    coeff_count,
                                    coeff_modulus[j],
                                    result + (j * coeff_count));
   }
}

/**
 * snippet for polynomial addition in NTT domain can be found at rlwe.cpp::encrypt_zero_asymmetric line 227-229
 * NTT (second CRT layer) memory layout is the same as the first layer
 * */
void add_dcrtpoly(util::ConstCoeffIter a, util::ConstCoeffIter b, std::size_t coeff_count,
                  std::vector<Modulus> const& coeff_modulus, util::CoeffIter result) {
#pragma omp parallel for
   for (size_t j = 0; j < coeff_modulus.size(); j++) {
      util::add_poly_coeffmod(a + (j * coeff_count),
                              b + (j * coeff_count),
                              coeff_count,
                              coeff_modulus[j],
                              result + (j * coeff_count));
   }
}

void sub_dcrtpoly(util::ConstCoeffIter a, util::ConstCoeffIter b, std::size_t coeff_count,
                  std::vector<Modulus> const& coeff_modulus, util::CoeffIter result) {
#pragma omp parallel for
   for (size_t j = 0; j < coeff_modulus.size(); j++) {
      util::sub_poly_coeffmod(a + (j * coeff_count),
                              b + (j * coeff_count),
                              coeff_count,
                              coeff_modulus[j],
                              result + (j * coeff_count));
   }
}

void assign_dcrtpoly(util::ConstCoeffIter a, std::size_t coeff_count, std::size_t coeff_modulus_count,
                     util::CoeffIter result) {
#pragma omp parallel for
   for (size_t i = 0; i < coeff_modulus_count; i++) {
      util::set_poly(a + (i * coeff_count), coeff_count, 1, result + (i * coeff_count));
   }
}

void to_eval_rep(util::CoeffIter a, size_t coeff_count, size_t coeff_modulus_count, util::NTTTables const* small_ntt_tables) {
#pragma omp parallel for
   for (size_t j = 0; j < coeff_modulus_count; j++) {
      util::ntt_negacyclic_harvey(a + (j * coeff_count), small_ntt_tables[j]); // ntt form
   }
}

/**
 * NOTE CKKSKeyRecovery & SEAL:
 *  this functions is the same as ckks.h::decode_internal line 687 to line 690
 *  (transform each polynomial from NTT domain into CRT)
 * */
void to_coeff_rep(util::CoeffIter a, size_t coeff_count, size_t coeff_modulus_count, util::NTTTables const* small_ntt_tables) {
#pragma omp parallel for
   for (size_t j = 0; j < coeff_modulus_count; j++) {
      util::inverse_ntt_negacyclic_harvey(a + (j * coeff_count), small_ntt_tables[j]); // non-ntt form
   }
}

/**
 * NOTE CKKSKeyRecovery:
 *  code below is similar to ckks.h::decode_internal line 692 to line 733
 *  the infty_norm of so-called "encoding error" is actually different from it in the paper
 *  with plaintext before decoding as m, re-encoded but not yet rounded plaintext as m*
 *  this project calculates the value of inf_norm(CastToDoubleCoeff(m - RoundToIntCoeff(m*)))
 *
 * NOTE SEAL:
 *  context_data.upper_half_threshold() == (product of all coeff modulus + 1) >> 1
 *  in ckks.h::decode_internal line 692 to line 733,
 *  coeffs of plain_copy are multi-precision integers, its memory is arranged in the following manner:
 *  (let d = coeff_modulus_size, n = coeff_count)
 *  [d uint64, from lsb to msb] ... [...] (the count of [...] is n)
 *  >> multi-precision memory layout <<
 *  -----------------------
 *  the double loop transforms the multi-precision integers into double
 *  the branch calculates (coeff - modulus) when coeff >= (1+modulus)/2, which transforms coeffs into [-(q-1)/2, (q+1)/2)
 *  -----------------------
 *  the coeff ranges from [-(q-1)/2, (q+1)/2), when representing it using a unsigned integer,
 *  the non-negative subrange is kept untouched, while the negative subrange is shifted to [(q+1)/2, q) by adding a q
 *  this representation is homomorphic wrt addition / multiplication / inverse (PROOF?)
 * */
long double infty_norm(util::ConstCoeffIter a, SEALContext::ContextData const* context_data) {
   auto &ciphertext_parms = context_data->parms();
   auto &coeff_modulus = ciphertext_parms.coeff_modulus();
   size_t coeff_mod_count = coeff_modulus.size();
   size_t coeff_count = ciphertext_parms.poly_modulus_degree();
   auto decryption_modulus = context_data->total_coeff_modulus();
   auto upper_half_threshold = context_data->upper_half_threshold();

   long double max = 0;

   auto aCopy(util::allocate_zero_poly(coeff_count, coeff_mod_count, MemoryManager::GetPool()));
   assign_dcrtpoly(a, coeff_count, coeff_mod_count, aCopy.get());

   // CRT-compose the polynomial
   context_data->rns_tool()->base_q()->compose_array(aCopy.get(), coeff_count, MemoryManager::GetPool());

   long double two_pow_64 = powl(2.0, 64);

   for (std::size_t i = 0; i < coeff_count; i++) {
      long double coeff = 0.0, cur_pow = 1.0;
      if (util::is_greater_than_or_equal_uint(aCopy.get() + (i * coeff_mod_count),
                                              upper_half_threshold, coeff_mod_count)) {
         for (std::size_t j = 0; j < coeff_mod_count; j++, cur_pow *= two_pow_64) {
            if (aCopy[i * coeff_mod_count + j] > decryption_modulus[j]) {
               auto diff = aCopy[i * coeff_mod_count + j] - decryption_modulus[j];
               coeff += diff ? static_cast<long double>(diff) * cur_pow : 0.0;
            } else {
               auto diff = decryption_modulus[j] - aCopy[i * coeff_mod_count + j];
               coeff -= diff ? static_cast<long double>(diff) * cur_pow : 0.0;
            }
         }
      } else {
         for (std::size_t j = 0; j < coeff_mod_count; j++, cur_pow *= two_pow_64) {
            auto curr_coeff = aCopy[i * coeff_mod_count + j];
            coeff += curr_coeff ? static_cast<long double>(curr_coeff) * cur_pow : 0.0;
         }
      }

      if (fabsl(coeff) > max) {
         max = fabsl(coeff);
      }
   }

   return max;
}

long double l2_norm(util::ConstCoeffIter a, SEALContext::ContextData const* context_data) {
   auto &ciphertext_parms = context_data->parms();
   auto &coeff_modulus = ciphertext_parms.coeff_modulus();
   size_t coeff_mod_count = coeff_modulus.size();
   size_t coeff_count = ciphertext_parms.poly_modulus_degree();
   auto decryption_modulus = context_data->total_coeff_modulus();
   auto upper_half_threshold = context_data->upper_half_threshold();

   long double sum = 0;

   auto aCopy(util::allocate_zero_poly(coeff_count, coeff_mod_count, MemoryManager::GetPool()));
   assign_dcrtpoly(a, coeff_count, coeff_mod_count, aCopy.get());

   // CRT-compose the polynomial
   context_data->rns_tool()->base_q()->compose_array(aCopy.get(), coeff_count, MemoryManager::GetPool());

   long double two_pow_64 = powl(2.0, 64);

   for (std::size_t i = 0; i < coeff_count; i++) {
      long double coeff = 0.0, cur_pow = 1.0;
      if (util::is_greater_than_or_equal_uint(aCopy.get() + (i * coeff_mod_count),
                                              upper_half_threshold, coeff_mod_count)) {
         for (std::size_t j = 0; j < coeff_mod_count; j++, cur_pow *= two_pow_64) {
            if (aCopy[i * coeff_mod_count + j] > decryption_modulus[j]) {
               auto diff = aCopy[i * coeff_mod_count + j] - decryption_modulus[j];
               coeff += diff ? static_cast<long double>(diff) * cur_pow : 0.0;
            } else {
               auto diff = decryption_modulus[j] - aCopy[i * coeff_mod_count + j];
               coeff -= diff ? static_cast<long double>(diff) * cur_pow : 0.0;
            }
         }
      } else {
         for (std::size_t j = 0; j < coeff_mod_count; j++, cur_pow *= two_pow_64) {
            auto curr_coeff = aCopy[i * coeff_mod_count + j];
            coeff += curr_coeff ? static_cast<long double>(curr_coeff) * cur_pow : 0.0;
         }
      }

      sum += coeff * coeff;
   }

   return sqrtl(sum);
}

std::string poly_to_string(std::uint64_t const* value, EncryptionParameters const& parms) {
   auto coeff_modulus = parms.coeff_modulus();
   size_t coeff_mod_count = coeff_modulus.size();
   size_t coeff_count = parms.poly_modulus_degree();
   std::ostringstream result;
   for (size_t i = 0; i < coeff_mod_count; i++) {
      auto mod = coeff_modulus[i].value();
      if (i>0) {
         result << std::endl;
      }
      result << "[" << mod << "]: ";
      for (size_t j = 0; j < coeff_count; j++) {
         std::uint64_t v = *value;
         if (v >= mod/2) {
            result << "-" << mod-v;
         } else {
            result << v;
         }
         result << (j==coeff_count?"":", ");
         value++;
      }
   }
   return result.str();
}


void print_poly(std::uint64_t const* value, EncryptionParameters const& parms, size_t max_count) {
   auto coeff_modulus = parms.coeff_modulus();
   size_t coeff_mod_count = coeff_modulus.size();
   size_t coeff_count = parms.poly_modulus_degree();
   for (size_t i = 0; i < coeff_mod_count; i++) {
      auto mod = coeff_modulus[i].value();
      std::uint64_t const* v = value + i*coeff_count;
      if (i>0) {
         std::cout << std::endl;
      }
      std::cout << "[" << mod << "]: ";
      for (size_t j = 0; j < coeff_count && (max_count == 0 || j < max_count); j++) {
         if (*v >= mod/2) {
            std::cout << "-" << mod-(*v);
         } else {
            std::cout << *v;
         }
         std::cout << (j==coeff_count?"":", ");
         v++;
      }
   }
   std::cout.flush();
}
