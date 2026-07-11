"""Regresioni testovi numeričkog modela."""

import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

import simulacija


class ParametriSistemaTest(unittest.TestCase):
    def test_izvedene_velicine_podrazumevanog_sistema(self) -> None:
        parametri = simulacija.PARAMETRI

        self.assertAlmostEqual(parametri.amplituda_sile, 200.0)
        self.assertAlmostEqual(parametri.sopstvena_frekvencija, 487.31050077104754)
        self.assertAlmostEqual(parametri.faktor_prigusenja, 0.005103103630798288)

    def test_nefizicki_parametri_se_odbacuju(self) -> None:
        neispravne_izmene = (
            {"masa": 0},
            {"krutost_opruge": -1},
            {"koeficijent_prigusenja": -1},
            {"povrsina": -1},
            {"pritisak": -1},
            {"frekvencija_pobude": -1},
        )

        for izmena in neispravne_izmene:
            with self.subTest(izmena=izmena), self.assertRaises(ValueError):
                replace(simulacija.PARAMETRI, **izmena)


class JednacinaTest(unittest.TestCase):
    def test_sila_dostize_amplitudu_na_cetvrtini_periode(self) -> None:
        cetvrtina_periode = 1 / (4 * simulacija.PARAMETRI.frekvencija_pobude)
        self.assertAlmostEqual(
            float(simulacija.sila(cetvrtina_periode)),
            simulacija.PARAMETRI.amplituda_sile,
        )

    def test_desna_strana_postuje_bilans_sila(self) -> None:
        izvod = simulacija.jednacina([0.001, 0.1], 0.25)
        ocekivano_ubrzanje = (200.0 - 0.1 - 300.0) / 0.032
        np.testing.assert_allclose(izvod, [0.1, ocekivano_ubrzanje])

    def test_stanje_mora_imati_dve_komponente(self) -> None:
        with self.assertRaisesRegex(ValueError, "tačno pomeranje i brzinu"):
            simulacija.jednacina([0.0], 0.0)


class ResavanjeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rezultat = simulacija.resi()

    def test_vremenska_osa_je_poluotvorena_i_ravnomerna(self) -> None:
        vreme = simulacija.vremenska_osa(1.0, 1.01, 1_000)

        self.assertEqual(vreme.size, 10)
        self.assertEqual(vreme[0], 1.0)
        self.assertLess(vreme[-1], 1.01)
        np.testing.assert_allclose(np.diff(vreme), 0.001, rtol=0, atol=1e-15)

    def test_vremenska_osa_odbacuje_korak_manji_od_float_preciznosti(self) -> None:
        with self.assertRaisesRegex(ValueError, "preciznost pokretnog zareza"):
            simulacija.vremenska_osa(1e16, 1e16 + 2, 1)

    def test_podrazumevano_resenje_je_konacno_i_pravilnog_oblika(self) -> None:
        self.assertEqual(self.rezultat.shape, (simulacija.t.size, 2))
        self.assertTrue(np.all(np.isfinite(self.rezultat)))
        np.testing.assert_array_equal(self.rezultat[0], simulacija.y0)

    def test_ustaljena_amplituda_odgovara_analitickoj(self) -> None:
        broj_odbiraka = simulacija.f_odabiranja
        vreme = simulacija.t[-broj_odbiraka:]
        pomeranje = self.rezultat[-broj_odbiraka:, 0]
        omega = simulacija.PARAMETRI.ugaona_frekvencija_pobude
        baza = np.column_stack(
            [np.sin(omega * vreme), np.cos(omega * vreme), np.ones_like(vreme)]
        )
        koeficijenti, *_ = np.linalg.lstsq(baza, pomeranje, rcond=None)
        numericka_amplituda = float(np.hypot(*koeficijenti[:2]))
        analiticka_amplituda = simulacija.PARAMETRI.amplituda_sile / np.hypot(
            simulacija.c - simulacija.M * omega**2,
            simulacija.u * omega,
        )

        np.testing.assert_allclose(
            numericka_amplituda,
            analiticka_amplituda,
            rtol=1e-7,
        )

    def test_ustaljeni_spektar_ima_samo_znacajan_vrh_na_jedan_herc(self) -> None:
        brzina = self.rezultat[-simulacija.f_odabiranja :, 1]
        amplituda = 2 * np.abs(np.fft.rfft(brzina)) / brzina.size
        amplituda[0] /= 2
        amplituda[-1] /= 2
        frekvencije = np.fft.rfftfreq(brzina.size, d=simulacija.T_odabiranja)
        dominantni_indeks = int(np.argmax(amplituda[1:]) + 1)

        self.assertEqual(frekvencije[dominantni_indeks], 1.0)
        ostale_amplitude = amplituda.copy()
        ostale_amplitude[[0, dominantni_indeks]] = 0
        self.assertLess(np.max(ostale_amplitude), amplituda[dominantni_indeks] * 1e-4)

    def test_sistem_bez_pobude_miruje_iz_nultog_stanja(self) -> None:
        parametri = replace(simulacija.PARAMETRI, pritisak=0)
        vreme = np.linspace(0, 0.01, 101)
        rezultat = simulacija.resi(vreme, parametri=parametri)

        np.testing.assert_array_equal(rezultat, np.zeros((vreme.size, 2)))

    def test_neispravna_vremenska_osa_se_odbacuje(self) -> None:
        neispravne_ose = ([0.0], [0.0, 0.0], [0.0, np.nan])
        for vreme in neispravne_ose:
            with self.subTest(vreme=vreme), self.assertRaises(ValueError):
                simulacija.resi(vreme)

    def test_neuspeh_integratora_se_prijavljuje(self) -> None:
        lazni_rezultat = np.zeros((2, 2))
        with mock.patch(
            "simulacija.odeint",
            return_value=(lazni_rezultat, {"message": "test greška"}),
        ), self.assertRaisesRegex(RuntimeError, "test greška"):
            simulacija.resi([0.0, 0.1])


if __name__ == "__main__":
    unittest.main()
