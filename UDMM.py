import numpy as np

class UDMM:
    def __init__(self, initial_model, initial_environment, plasticity=0.1, lambda_=0.5, emotion_factor=0.3):
        """
        مُعلمة التأسيس:
        - initial_model: توزيع احتمالي مُبتدئ للنموذج الداخلي (مصفوفة numpy).
        - initial_environment: توزيع احتمالي مُبتدئ للبيئة (مصفوفة numpy بنفس أبعاد النموذج).
        - plasticity: معامل اللدونة العصبية (الافتراضي: 0.1).
        - lambda_: معامل الموازنة بين المعلومات والإنتروبيا (الافتراضي: 0.5).
        - emotion_factor: تأثير المشاعر على اللدونة (الافتراضي: 0.3).
        """
        self.model = np.array(initial_model, dtype=float)
        self.environment = np.array(initial_environment, dtype=float)
        self.plasticity = plasticity
        self.lambda_ = lambda_
        self.emotion_factor = emotion_factor
        self.validate_probability_distributions()

    def validate_probability_distributions(self):
        """تحقق من صحة التوزيعات الاحتمالية (مجموع 1 والأبعاد متطابقة)."""
        assert np.allclose(np.sum(self.model), 1), "التوزيع الداخلي غير صحيح (المجموع غير 1)."
        assert np.allclose(np.sum(self.environment), 1), "التوزيع البيئي غير صحيح (المجموع غير 1)."
        assert self.model.shape == self.environment.shape, "الأبعاد غير متطابقة."

    def entropy(self, distribution):
        """حساب الإنتروبيا لـ توزيع احتمالي."""
        return -np.sum(distribution * np.log(distribution + 1e-9))

    def kl_divergence(self, p, q):
        """حساب الإنتروبيا الكونية (Kullback-Leibler) بين p و q."""
        return np.sum(p * np.log((p + 1e-9) / (q + 1e-9)))

    def mutual_information(self):
        """حساب المعلومات المتبادلة بين النموذج والبيئة."""
        h_model = self.entropy(self.model)
        h_environment = self.entropy(self.environment)
        h_joint = self.entropy(self.model * self.environment)  # افتراض الاستقلالية
        return h_model + h_environment - h_joint

    def dynamic_plasticity(self, emotion_intensity):
        """حساب اللدونة الديناميكية بناءً على المشاعر."""
        return self.plasticity * (1 + self.emotion_factor * emotion_intensity)

    def update_model(self, sensory_input, disturbance=0, emotion_intensity=0):
        """تحديث النموذج الداخلي بناءً على المدخلات الحسية."""
        # 1. حساب الخطأ التنبؤي
        prediction_error = self.kl_divergence(sensory_input, self.model)

        # 2. حساب التكيف بناءً على المعلومات والإنتروبيا
        mi = self.mutual_information()
        adjustment = (self.dynamic_plasticity(emotion_intensity) *
                      (sensory_input - self.model) -
                      self.lambda_ * prediction_error)

        # 3. تحديث النموذج مع الإضطرابات
        self.model += adjustment + disturbance
        self.model = np.clip(self.model, 1e-9, 1)  # تجنب قيم غير صالحة
        self.model /= np.sum(self.model)  # إعادة التطبيع

        return self.model, prediction_error

    def adapt(self, sensory_input, disturbance=0, emotion_intensity=0):
        """تنفيذ التكيف مع البيئة مع إرجاع نتائج التحليل."""
        updated_model, error = self.update_model(sensory_input, disturbance, emotion_intensity)
        return {
            'model': updated_model,
            'error': error,
            'mutual_information': self.mutual_information(),
            'entropy': self.entropy(self.model),
            'environment_entropy': self.entropy(self.environment),
        }

    def special_case_analysis(self):
        """تحليل حالة الاستقرار أو التكيف."""
        if np.allclose(self.model, self.environment):
            return "النظام في حالة استقرار."
        return "النظام يتكيف مع التغيرات."

    def error_analysis(self, sensory_input):
        """تحليل الخطأ باستخدام مبدأ جينسن."""
        h_model = self.entropy(self.model)
        h_sensory = self.entropy(sensory_input)
        return {
            'KL_Divergence': self.kl_divergence(sensory_input, self.model),
            'Upper_Bound': np.sqrt((h_model + h_sensory) / 2)  # حد أقصى للخطأ
        }

# ----------------------- مثال تطبيقي -----------------------

# توزيعات احتمالية مثال (يجب أن يكون مجموع كل منها 1)
initial_model = np.array([0.2, 0.3, 0.5])
initial_environment = np.array([0.4, 0.4, 0.2])

# إنشاء مثال من الفصل UDMM
udmm = UDMM(
    initial_model=initial_model,
    initial_environment=initial_environment,
    plasticity=0.15,       # زيادة اللدونة العصبية
    lambda_=0.3,           # تقليل تأثير الإنتروبيا
    emotion_factor=0.4     # تأثير عاطفي عالٍ
)

# مدخلات حسية جديدة (مثال: بيئة مُتغير)
new_sensory_input = np.array([0.6, 0.2, 0.2])

# تطبيق التكيف مع إضطراب بسيط وتاثير عاطفي
result = udmm.adapt(
    sensory_input=new_sensory_input,
    disturbance=0.05,        # إضطراب خارجي
    emotion_intensity=1.5    # حالة عاطفية (مثل: فضول)
)

# عرض النتائج
print("النموذج بعد التكيف:", result['model'].round(2))
print("الخطأ التنبؤي:", round(result['error'], 4))
print("المعلومات المتبادلة:", round(result['mutual_information'], 4))
print("الإنتروبيا بعد التكيف:", round(result['entropy'], 4))
print("تحليل الحالة:", udmm.special_case_analysis())
print("تحليل الخطأ:", udmm.error_analysis(new_sensory_input))
