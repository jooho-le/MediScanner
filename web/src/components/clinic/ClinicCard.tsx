interface Clinic {
  name: string;
  specialty: string;
  address: string;
  distance: string;
}

interface ClinicCardProps {
  clinic: Clinic;
}

const ClinicCard = ({ clinic }: ClinicCardProps) => (
  <div className="clinic-card">
    <div>
      <h3>{clinic.name}</h3>
      <p>{clinic.address}</p>
      <span>{clinic.specialty}</span>
    </div>
    <button>지도 보기 · {clinic.distance}</button>
  </div>
);

export default ClinicCard;
