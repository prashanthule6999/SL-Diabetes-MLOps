import unittest
from fastapi.testclient import TestClient
from fastapp.app import app


class FastAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "<title>Diabetes Prediction System</title>",
            response.text
        )

    def test_predict_page(self):
        response = self.client.post(
            "/predict",
            json={
                "pregnancies": 6,
                "glucose": 148,
                "blood_pressure": 72,
                "skin_thickness": 35,
                "insulin": 1,
                "bmi": 33.6,
                "diabetes_pedigree_function": 0.627,
                "age": 50
            }
        )

        self.assertEqual(response.status_code, 200)

        result = response.json()
        self.assertIn("prediction", result)
        self.assertIn("confidence", result)
        self.assertIn("class_probabilities", result)

        self.assertIn(result["prediction"], [0, 1])

        self.assertTrue(0 <= result["confidence"] <= 1)

        probs = result["class_probabilities"]
        self.assertIsInstance(probs, dict)

        for p in probs.values():
            self.assertTrue(0 <= p <= 1)

        self.assertAlmostEqual(sum(probs.values()), 1.0, places=4)

if __name__ == "__main__":
    unittest.main()
