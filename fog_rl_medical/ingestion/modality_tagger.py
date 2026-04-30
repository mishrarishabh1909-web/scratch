class ModalityTagger:
    def tag(self, task):
        modality = task.get('modality', '').upper()
        if modality in ['ECG', 'IMAGE', 'VITALS', 'TEXT']:
            task['tagged_modality'] = modality
        else:
            task['tagged_modality'] = 'UNKNOWN'
        return task
